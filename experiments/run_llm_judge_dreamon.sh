#!/bin/bash
#SBATCH --job-name=lj_dreamon
#SBATCH --output=logs/lj_dreamon_%j.out
#SBATCH --error=logs/lj_dreamon_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research-eemcs-st

# ============================================================
# LLM-as-Judge: DreamOn results, judged by Qwen2.5-7B-Instruct
# Wraps experiments/llm_judge_variable_naming.py and runs it on
# the three DreamOn variants (single_window, tiled_first_mask,
# tiled_majority).
#
# Usage:
#   sbatch experiments/run_llm_judge_dreamon.sh
#   sbatch experiments/run_llm_judge_dreamon.sh \
#       --judge-model Qwen2.5-14B-Instruct
#
# On Colab (no SLURM):
#   bash experiments/run_llm_judge_dreamon.sh
# ============================================================

set -euo pipefail

# ---------- Paths ----------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

# ---------- DelftBlue module loading (skipped on Colab) -------------------
if command -v module &>/dev/null; then
    module purge
    module load 2023r1
    module load python/3.11.1
    module load cuda/11.8
fi

# ---------- Activate venv (skipped if no .venv on Colab) ------------------
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$PROJECT_DIR"

# ---------- Settings -------------------------------------------------------
JUDGE_MODEL="${JUDGE_MODEL:-Qwen2.5-7B-Instruct}"
MAX_SAMPLES_ARG=""
if [ -n "${MAX_SAMPLES:-}" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

# Allow offline mode on DelftBlue (skipped if not set)
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p logs results/llm_judge_dreamon

# ---------- Step 1: build baseline-format CSVs from DreamOn outputs ------
echo "============================================================"
echo "  Step 1: convert DreamOn CSVs -> baseline format"
echo "============================================================"
python3 experiments/prepare_dreamon_for_judge.py

# ---------- Step 2: run the judge on each variant -------------------------
DREAMON_DIR="data/benchmark_ReFineID_DreamOn"
INPUTS=(
    "$DREAMON_DIR/DreamOn-7B_single_window.csv"
    "$DREAMON_DIR/DreamOn-7B_tiled_first_mask.csv"
    "$DREAMON_DIR/DreamOn-7B_tiled_majority.csv"
)

echo ""
echo "============================================================"
echo "  Step 2: LLM-as-Judge with $JUDGE_MODEL"
echo "============================================================"

for INPUT in "${INPUTS[@]}"; do
    if [ ! -f "$INPUT" ]; then
        echo "SKIP (missing): $INPUT"
        continue
    fi
    echo ""
    echo "----------------------------------------------------------"
    echo "  Judging: $INPUT"
    echo "----------------------------------------------------------"
    python3 experiments/llm_judge_variable_naming.py \
        --input "$INPUT" \
        --judge-model "$JUDGE_MODEL" \
        --results-dir results/llm_judge_dreamon \
        --resume \
        $MAX_SAMPLES_ARG \
        "$@"
done

echo ""
echo "============================================================"
echo "  Done. Outputs in results/llm_judge_dreamon/"
echo "============================================================"
ls -1 results/llm_judge_dreamon/
