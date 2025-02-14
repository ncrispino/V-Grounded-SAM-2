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

"""
Step 1: Environment settings and model initialization
"""
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

def main(video_dir, output_dir, output_video_path, text, prompt_type="box", box_threshold=0.25, text_threshold=0.3):
    """Run SAM 2 video predictor and Grounding DINO to track objects in video frames.
    
    Args:
        video_dir (str): Directory containing JPEG frames with filenames like `<frame_index>.jpg`
        output_dir (str): Directory to save the annotated frames
        output_video_path (str): Path to save the final video
        text (str): Text prompt for Grounding DINO (must be lowercased and end with a dot)
        prompt_type (str): Type of prompt to use ("point", "box", or "mask")
        box_threshold (float): Threshold for box detection
        text_threshold (float): Threshold for text detection

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
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(result_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    """
    Step 6: Create final video
    """
    create_video_from_images(result_dir, output_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="The directory of JPEG frames with filenames like `<frame_index>.jpg`")
    parser.add_argument("--text", type=str, required=True, help="The text prompt for Grounding DINO. Must be lowercased and end with a dot.")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="The threshold for box detection")
    parser.add_argument("--text_threshold", type=float, default=0.3, help="The threshold for text detection")
    args = parser.parse_args()
    
    output_dir = args.video_dir + f"/../gsam2_frames_{args.box_threshold}_{args.text_threshold}"  # The directory to save the annotated frames
    output_video_path = args.video_dir + f"/../gsam2_{args.box_threshold}_{args.text_threshold}.mp4"  # The path to save the final video
    video_dir = os.path.join(args.video_dir, '../../frames')
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output video path: {output_video_path}")
    
    main(
        video_dir=video_dir,
        output_dir=output_dir,
        output_video_path=output_video_path,
        text=args.text,
        prompt_type="box",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
