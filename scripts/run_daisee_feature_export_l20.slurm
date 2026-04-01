#!/bin/bash
#SBATCH -J daisee_feat_export
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o logs/daisee_feature_export_%j.out

# EDIT ONLY: USERNAME, ENV_NAME
source /public/home/USERNAME/miniconda3/bin/activate ENV_NAME
cd /public/home/USERNAME/k12_stream_state_v1

python -m src.export.export_daisee_features \
  --manifest_csv data/interim/daisee_manifest.csv \
  --dataset_root /public/home/USERNAME/datasets/DAiSEE \
  --out_dir data/interim/daisee_proxy_features \
  --out_manifest_csv data/interim/daisee_manifest_with_features.csv \
  --num_frames 32 --image_size 160
