#!/bin/bash
#SBATCH -J daisee_train
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/daisee_train_%j.out

# EDIT ONLY: USERNAME, ENV_NAME
source /public/home/USERNAME/miniconda3/bin/activate ENV_NAME
cd /public/home/USERNAME/k12_stream_state_v1

python -m src.training.train_daisee --config configs/daisee.yaml
