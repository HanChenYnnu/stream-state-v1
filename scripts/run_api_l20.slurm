#!/bin/bash
#SBATCH -J k12_api
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH -o logs/api_%j.out

source /public/home/USERNAME/miniconda3/bin/activate k12stream
cd /public/home/USERNAME/k12_stream_state_v1

uvicorn src.api.app:app --host 0.0.0.0 --port 8000
