#!/bin/bash
#SBATCH -J daisee_eval
#SBATCH -p GPU_L20
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/daisee_eval_%j.out

# EDIT ONLY: USERNAME, ENV_NAME
source /public/home/USERNAME/miniconda3/bin/activate ENV_NAME
cd /public/home/USERNAME/k12_stream_state_v1

python -m src.training.eval_daisee --config configs/daisee.yaml --checkpoint results/daisee/train_run/best.pt
python -m src.mapping.daisee_to_learning_state --pred_csv results/daisee/eval_run/predictions.csv --out_csv results/daisee/eval_run/mapped_states.csv
