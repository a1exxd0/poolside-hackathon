#!/bin/bash
#SBATCH --job-name=laguna-xs2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

# Cache model weights to scratch to avoid filling home quota
export HF_HOME=${EPHEMERAL:-$HOME}/.cache/huggingface

mkdir -p logs

source venv/bin/activate
python load_locally.py
