#!/bin/bash
#SBATCH -J k12_dual_fusion
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/dual_fusion_%j.out

source /public/home/USERNAME/miniconda3/bin/activate k12stream
cd /public/home/USERNAME/k12_stream_state_v1

# Template: first task runs visual extraction, second runs downstream fusion/training placeholder.
srun --exclusive -N1 -n1 python -m src.pipelines.infer_video --input data/raw/demo_videos/sample.mp4 --out_video data/outputs/videos/fusion.mp4 --out_csv data/outputs/csv/fusion.csv &
srun --exclusive -N1 -n1 python -c "print('TODO: add temporal fusion trainer in phase-2')" &
wait
