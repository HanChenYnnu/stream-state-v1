#!/bin/bash
#SBATCH -J daisee_manifest
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/daisee_manifest_%j.out

# EDIT ONLY: USERNAME, ENV_NAME
source /public/home/USERNAME/miniconda3/bin/activate ENV_NAME
cd /public/home/USERNAME/k12_stream_state_v1

python -m src.datasets.daisee_manifest \
  --dataset_root /public/home/USERNAME/datasets/DAiSEE \
  --out_csv data/interim/daisee_manifest.csv
