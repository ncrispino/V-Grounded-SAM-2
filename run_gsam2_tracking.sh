# Usage: bash run_gsam2_tracking_w_cont_ids.sh <path_to_video> <text_file_name>
# Run the following in the gsam2_env virtual environment.
# python grounded_sam2_tracking_demo_with_continuous_id.py --video_dir=$1 --text=$2

# echo "video_dir: $1"
# echo "text: $2"
# echo "box_threshold: $3"
# echo "text_threshold: $4"
# Old
# ./gsam2_env/bin/python grounded_sam2_tracking_demo.py --video_dir=$1 --text="$2" --box_threshold=$3 --text_threshold=$4
#
# New (2/13/2025)
echo "config_path: $1"
echo "n_gpus: $2"
./gsam2_env/bin/python grounded_sam2_tracking_demo_multiprocess.py --config_path $1 --n_gpus $2
