#!/bin/bash
#SBATCH --job-name=llm_judge
#SBATCH --output=logs/llm_judge_%j.out
#SBATCH --error=logs/llm_judge_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research-eemcs-st

# ============================================================
# LLM-as-a-Judge: Variable Renaming Evaluation
# Runs on DelftBlue GPU cluster
#
# Usage:
#   sbatch experiments/run_llm_judge.sh
#   sbatch experiments/run_llm_judge.sh --judge-model Qwen2.5-14B-Instruct
#   sbatch experiments/run_llm_judge.sh --max-samples 100  # quick test
#
# Environment variables you can set before sbatch:
#   JUDGE_MODEL  - judge model name (default: Qwen2.5-7B-Instruct)
#   MAX_SAMPLES  - max samples per file (default: all)
# ============================================================

set -euo pipefail

# ---------- Paths -----------------------------------------------------------
PROJECT_DIR="$HOME/CoRefusion"
VENV_DIR="$PROJECT_DIR/.venv"

# ---------- Module loading --------------------------------------------------
module purge
module load 2023r1
module load python/3.11.1
module load cuda/11.8

# ---------- Activate virtualenv or install deps ----------------------------
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch transformers accelerate tqdm pandas
fi

# ---------- Settings (can be overridden via environment variables) ----------
JUDGE_MODEL="${JUDGE_MODEL:-Qwen2.5-7B-Instruct}"
MAX_SAMPLES_ARG=""
if [ -n "${MAX_SAMPLES:-}" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

# Offline mode: models should already be cached in $HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ---------- Run judge -------------------------------------------------------
echo "============================================================"
echo "  LLM Judge: Variable Renaming Evaluation"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Node         : $(hostname)"
echo "  Judge model  : $JUDGE_MODEL"
echo "  Start        : $(date)"
echo "============================================================"

mkdir -p logs

cd "$PROJECT_DIR"
python3 experiments/llm_judge_variable_naming.py \
    --all \
    --judge-model "$JUDGE_MODEL" \
    --resume \
    $MAX_SAMPLES_ARG \
    "$@"

echo ""
echo "============================================================"
echo "  Finished at: $(date)"
echo "============================================================"
