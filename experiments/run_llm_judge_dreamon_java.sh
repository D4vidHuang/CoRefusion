#!/bin/bash
#SBATCH --job-name=lj_dreamon_java
#SBATCH --output=logs/lj_dreamon_java_%j.out
#SBATCH --error=logs/lj_dreamon_java_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=research-eemcs-st

# ============================================================
# LLM-as-Judge for the AISE DreamOn Java-Identifier models,
# judged by Qwen2.5-7B-Instruct.
#
# Wraps:
#   1) experiments/prepare_dreamon_java_for_judge.py   (per_sample -> baseline)
#   2) experiments/llm_judge_variable_naming.py        (the judge)
#
# Run the EM benchmark FIRST so results/dreamon_java/*_per_sample_*.csv exist:
#   python experiments/benchmark_dreamon_java_identifiers.py
#
# Usage:
#   sbatch experiments/run_llm_judge_dreamon_java.sh
#   sbatch experiments/run_llm_judge_dreamon_java.sh --judge-model Qwen2.5-14B-Instruct
#   # On Colab (no SLURM):
#   bash   experiments/run_llm_judge_dreamon_java.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

# ---------- DelftBlue modules (skipped on Colab) --------------------------
if command -v module &>/dev/null; then
    module purge
    module load 2023r1
    module load python/3.11.1
    module load cuda/11.8
fi
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi
cd "$PROJECT_DIR"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen2.5-7B-Instruct}"
MAX_SAMPLES_ARG=""
if [ -n "${MAX_SAMPLES:-}" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p logs results/llm_judge_dreamon_java

# ---------- Step 1: build baseline-format CSVs ----------------------------
echo "============================================================"
echo "  Step 1: convert DreamOn-Java per_sample CSVs -> baseline format"
echo "============================================================"
python3 experiments/prepare_dreamon_java_for_judge.py

# ---------- Step 2: judge every produced variant --------------------------
JDIR="data/benchmark_ReFineID_DreamOn_Java"
echo ""
echo "============================================================"
echo "  Step 2: LLM-as-Judge with $JUDGE_MODEL"
echo "============================================================"

shopt -s nullglob
INPUTS=("$JDIR"/*.csv)
if [ ${#INPUTS[@]} -eq 0 ]; then
    echo "ERROR: no CSVs in $JDIR (did Step 1 run?)."
    exit 1
fi

for INPUT in "${INPUTS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "  Judging: $INPUT"
    echo "----------------------------------------------------------"
    python3 experiments/llm_judge_variable_naming.py \
        --input "$INPUT" \
        --judge-model "$JUDGE_MODEL" \
        --results-dir results/llm_judge_dreamon_java \
        --resume \
        $MAX_SAMPLES_ARG \
        "$@"
done

echo ""
echo "============================================================"
echo "  Done. Outputs in results/llm_judge_dreamon_java/"
echo "============================================================"
ls -1 results/llm_judge_dreamon_java/
