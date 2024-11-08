# Usage: bash run_gsam2_tracking_w_cont_ids.sh <path_to_video> <text_file_name>
# Run the following in the gsam2_env virtual environment.
# python grounded_sam2_tracking_demo_with_continuous_id.py --video_dir=$1 --text=$2

./gsam2_env/bin/python grounded_sam2_tracking_demo_with_continuous_id.py --video_dir=$1 --text=$2
