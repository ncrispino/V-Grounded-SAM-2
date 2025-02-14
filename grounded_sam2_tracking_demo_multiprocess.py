import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import argparse
import json
import multiprocessing
from tqdm import tqdm


def initialize_model(gpu_id):
    """
    Step 1: Environment settings and model initialization
    """
    torch.cuda.set_device(int(gpu_id))
    print(f"Initializing worker on GPU {gpu_id}")

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return video_predictor, image_predictor, processor, grounding_model, device

def main(video_predictor, image_predictor, processor, grounding_model, device, video_dir, output_dir, output_video_path, text, prompt_type="box", box_threshold=0.25, text_threshold=0.3, include_mask_in_video=False):
    """Run SAM 2 video predictor and Grounding DINO to track objects in video frames.
    
    Args:
        video_dir (str): Directory containing JPEG frames with filenames like `<frame_index>.jpg`
        output_dir (str): Directory to save the annotated frames
        output_video_path (str): Path to save the final video
        text (str): Text prompt for Grounding DINO (must be lowercased and end with a dot)
        prompt_type (str): Type of prompt to use ("point", "box", or "mask")
        box_threshold (float): Threshold for box detection
        text_threshold (float): Threshold for text detection
        include_mask_in_video (bool): Whether to include masks in the final video. Default is False bc they will mess up certain attributes, e.g., colors (if we have yellow mask on red object, it will screw up count, spatial questions).

    Returns:
        None
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    json_data_dir = os.path.join(output_dir, "json_data")
    mask_data_dir = os.path.join(output_dir, "mask_data")
    result_dir = os.path.join(output_dir, "result")
    os.makedirs(json_data_dir, exist_ok=True)
    os.makedirs(mask_data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Scan all frame names in the directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir)
    ann_frame_idx = 0  # the frame index we interact with

    """
    Step 2: Prompt Grounding DINO and SAM image predictor
    """
    img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # Run Grounding DINO
    text = text.strip('"\'')
    print(f"Prompting Grounding DINO with text: {text}")
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )

    # Get masks from SAM image predictor
    image_predictor.set_image(np.array(image.convert("RGB")))
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Convert mask shape
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register objects with video predictor
    """
    if prompt_type == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif prompt_type == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif prompt_type == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

    """
    Step 4: Propagate through video
    """
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Save data and create annotated frames
    """
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
        image_base_name = os.path.splitext(frame_names[frame_idx])[0]
        
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        
        # Get bounding boxes from masks
        boxes = sv.mask_to_xyxy(masks)
        
        # Save mask data
        if len(masks) > 0:
            mask_array = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            for idx, (obj_id, mask) in enumerate(zip(object_ids, masks)):
                mask_array[mask] = obj_id
            np.save(os.path.join(mask_data_dir, f"mask_{image_base_name}.npy"), mask_array)
        
        # Save JSON data
        json_data = {
            "frame_index": frame_idx,
            "labels": {},
            "mask_name": f"mask_{image_base_name}.npy",
            "mask_height": img.shape[0],
            "mask_width": img.shape[1]
        }
        
        for idx, obj_id in enumerate(object_ids):
            box = boxes[idx]
            json_data["labels"][str(obj_id)] = {
                "instance_id": int(obj_id),
                "class_name": ID_TO_OBJECTS[obj_id],
                "box": box.tolist(),
                "score": float(scores[idx]) if scores is not None else None
            }
        
        with open(os.path.join(json_data_dir, f"mask_{image_base_name}.json"), 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Create annotated frame
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        if include_mask_in_video:
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(result_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    """
    Step 6: Create final video
    """
    create_video_from_images(result_dir, output_video_path)

def process_config(config, video_predictor, image_predictor, processor, grounding_model, device):
    """
    Process one configuration with the provided model.
    Replace the dummy processing with your actual work.
    """
    try:
        if os.path.exists(config["output_video_path"]):
            return {}  # bail out if the output video already exists
        main(
            video_predictor, image_predictor, processor, grounding_model, device,
            video_dir=config["video_dir"],
            output_dir=config["output_dir"],
            output_video_path=config["output_video_path"],
            text=config["text"],
            prompt_type=config.get("prompt_type", "box"),
            box_threshold=config.get("box_threshold", 0.10),
            text_threshold=config.get("text_threshold", 0.15)
        )
        result_output = f"Processed {config['video_dir']} using Grounding SAM 2 and saved to {config['output_video_path']}"
        status = "success"
    except Exception as e:
        result_output = str(e)
        status = "failed"

    job_result = {
        "config": config,
        "status": status,
        "result": result_output
    }
    return job_result

def process_configs_on_gpu(gpu_id, configs, results_list):
    """
    This function runs in a separate process. It sets the appropriate GPU,
    initializes the model once, and then processes each config in its own loop.
    Appends the results to results_list (a Manager list).
    """
    # Initialize the model once.
    video_predictor, image_predictor, processor, grounding_model, device = initialize_model(gpu_id)
    local_results = []
    for config in tqdm(configs, desc=f"GPU {gpu_id}", leave=False):
        r = process_config(config, video_predictor, image_predictor, processor, grounding_model, device)
        local_results.append(r)
    # Append results to the shared list.
    results_list += local_results

def generate_from_list_and_collect_results(config_file_path, n_gpus, result_file_path="completed_jobs.json"):
    # Ensure that the multiprocessing context is set to 'spawn' for PyTorch.
    multiprocessing.set_start_method('spawn', force=True)

    with open(config_file_path, 'r') as f:
        config_items = json.load(f)

    # Optionally, limit the dataset for testing:
    # config_items = [d for d in config_items if d["task_name"] == "ObjectRecognition"][:1] + [d for d in config_items if d["task_name"] == "AttributeRecognition"][:1] + [d for d in config_items if d["task_name"] == "SpatialUnderstanding"][:1] + [d for d in config_items if d["task_name"] == "Counting"][:1] + [d for d in config_items if d["task_name"] == "ActionRecognition"][:1] + [d for d in config_items if d["task_name"] == "SceneUnderstanding"][:1]

    # Assume each item is {"config": { ... }}, adjust if your JSON has a different structure.
    config_list = [item["config"] for item in config_items]

    print(f"Processing {len(config_list)} configs")
    # config_list = config_list[:4]

    # Determine available GPUs. For example, check the env variable or use torch.
    all_gpu_ids_env = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if all_gpu_ids_env:
        all_gpu_ids = all_gpu_ids_env.split(",")
    else:
        all_gpu_ids = [str(i) for i in range(torch.cuda.device_count())]

    if len(all_gpu_ids) != n_gpus:
        raise ValueError(f"Found only {len(all_gpu_ids)} GPUs, but n_gpus was specified as {n_gpus}")

    # selected_gpu_ids = all_gpu_ids[:n_gpus]
    print(f"Using Node GPUs: {all_gpu_ids}")
    selected_gpu_ids = range(n_gpus)  # bc we already requested desired GPUs with CUDA_VISIBLE_DEVICES
    print(f"Using GPUs: {selected_gpu_ids}")

    # Split the configuration list into n_gpus chunks
    chunks = [[] for _ in range(n_gpus)]
    for i, config in enumerate(config_list):
        chunks[i % n_gpus].append(config)

    manager = multiprocessing.Manager()
    results_list = manager.list()

    processes = []
    for idx, gpu_id in enumerate(selected_gpu_ids):
        configs_for_gpu = chunks[idx]
        p = multiprocessing.Process(target=process_configs_on_gpu, args=(gpu_id, configs_for_gpu, results_list))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Convert shared list to a normal list.
    all_results = list(results_list)

    # now join back with instances based on config.
    matched_results = []
    for config in config_list:
        for item in config_items:
            if item["config"] == config:
                matched_results.append(item)
                break
    with open(result_file_path, 'w') as f:
        json.dump(matched_results, f, indent=2)
    print(f"Saved aggregated job results to {result_file_path}")

# def run_config(args):
#     # Unpack tuple of (config, gpu_id)
#     config, gpu_id = args

#     # Use torch.cuda.set_device to pick the proper GPU for this worker.
#     torch.cuda.set_device(int(gpu_id))
#     print(f"Starting job on GPU {gpu_id} with config: {config}")

#     try:
#         # Call your main processing function.
#         result_output = main(
#             video_dir=config["video_dir"],
#             output_dir=config["output_dir"],
#             output_video_path=config["output_video_path"],
#             text=config["text"],
#             prompt_type=config.get("prompt_type", "box"),
#             box_threshold=config.get("box_threshold", 0.10),
#             text_threshold=config.get("text_threshold", 0.15)
#         )
#         status = "success"
#     except Exception as e:
#         result_output = str(e)
#         status = "failed"

#     job_result = {
#         "gpu": gpu_id,
#         "config": config,
#         "status": status,
#         "result": result_output
#     }
#     print(f"Finished job on GPU {gpu_id} with result: {job_result}")
#     return job_result


# def generate_from_list_and_collect_results(config_file_path, n_gpus, result_file_path="completed_jobs.json"):
#     """
#     Launches a job for each config concurrently over n_gpus, explicitly setting GPUs using torch.cuda.set_device,
#     and collects results into a JSON file.
#     """
#     # Ensure that the multiprocessing context is set to 'spawn' for PyTorch.
#     multiprocessing.set_start_method('spawn', force=True)

#     with open(config_file_path, 'r') as f:
#         config_items = json.load(f)
#     # If your json file encloses each config under a "config" key, use:
#     config_list = [item["config"] for item in config_items][:8]  # limit to first 8 for testing
#     # Otherwise if it is a list of dicts, you can simply use:
#     # config_list = config_items

#     # Determine available GPUs. We expect CUDA_VISIBLE_DEVICES to list the device ids (as strings).
#     all_gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES", "")
#     if all_gpu_ids:
#         all_gpu_ids = all_gpu_ids.split(",")
#     else:
#         # Fallback to torch device count.
#         all_gpu_ids = [str(i) for i in range(torch.cuda.device_count())]

#     if len(all_gpu_ids) != n_gpus:
#         raise ValueError(f"Found only {len(all_gpu_ids)} GPUs, but n_gpus was specified as {n_gpus}")

#     print(f"Using GPUs: {all_gpu_ids[:n_gpus]}")

#     # Create argument list, assigning configs to GPUs in round-robin fashion.
#     args_list = []
#     for i, config in enumerate(config_list):
#         gpu_id = all_gpu_ids[i % n_gpus]
#         args_list.append((config, gpu_id))

#     all_results = []
#     # Use a multiprocessing Pool with a number of processes equal to n_gpus.
#     with multiprocessing.Pool(processes=n_gpus) as pool:
#         # Use imap_unordered with a progress bar.
#         for job_result in tqdm(pool.imap_unordered(run_config, args_list), total=len(args_list)):
#             all_results.append(job_result)

#     # Write out the results.
#     with open(result_file_path, 'w') as f:
#         json.dump(all_results, f, indent=2)
#     print(f"Saved aggregated job results to {result_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to JSON config file", required=True)
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to run concurrently")
    args = parser.parse_args()

    result_file = args.config_path.replace(".json", "_results.json")
    generate_from_list_and_collect_results(args.config_path, args.n_gpus, result_file)
