#!/bin/bash
#SBATCH -J k12_dual_vision
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/dual_vision_%j.out

source /public/home/USERNAME/miniconda3/bin/activate k12stream
cd /public/home/USERNAME/k12_stream_state_v1

srun --exclusive -N1 -n1 python -m src.pipelines.infer_video --input data/raw/demo_videos/sample_a.mp4 --out_video data/outputs/videos/a.mp4 --out_csv data/outputs/csv/a.csv &
srun --exclusive -N1 -n1 python -m src.pipelines.infer_video --input data/raw/demo_videos/sample_b.mp4 --out_video data/outputs/videos/b.mp4 --out_csv data/outputs/csv/b.csv &
wait
