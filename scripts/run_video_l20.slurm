#!/bin/bash
#SBATCH -J k12_video_infer
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/video_%j.out

source /public/home/USERNAME/miniconda3/bin/activate k12stream
cd /public/home/USERNAME/k12_stream_state_v1

python -m src.pipelines.infer_video \
  --input data/raw/demo_videos/sample.mp4 \
  --out_video data/outputs/videos/demo_overlay.mp4 \
  --out_csv data/outputs/csv/demo_states.csv \
  --out_figure data/outputs/figures/demo_timeline.png
