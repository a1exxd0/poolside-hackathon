#!/bin/bash
#SBATCH --job-name=laguna-benchmark
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail

# ── HuggingFace auth ──────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not found in .env or environment." && exit 1
fi

# Cache model weights to scratch/ephemeral to avoid filling home quota
export HF_HOME=${EPHEMERAL:-$HOME}/.cache/huggingface

mkdir -p logs results

source venv/bin/activate

# Install all dependencies (idempotent — skips already-installed packages)
pip install -q -r requirements.txt

python benchmark.py \
    --model poolside/Laguna-XS.2 \
    --budget 1024 \
    --humaneval \
    --livecodebench \
    --longbench \
    --longbench-tasks lcc repobench-p 2wikimqa hotpotqa \
    --n 100 \
    --out results/benchmark_${SLURM_JOB_ID}.json
