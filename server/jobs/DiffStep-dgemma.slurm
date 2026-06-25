#!/bin/bash
#SBATCH --job-name=diffstep_dgemma
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/diffstep-dgemma-%j.out
#SBATCH --error=logs/diffstep-dgemma-%j.err
# Diffusion-step sweep for DiffusionGemma-26B-A4B (Fig 6, 3rd line).
# Disables adaptive early stopping and sets max-denoising-steps = T over the grid,
# so it always runs all steps -- the block-AR analogue of the DiffuCoder/DreamCoder
# diffusion_generate(steps=T) sweep.
#
# !! bf16 weights ~52GB -> needs the 96GB RTX PRO 6000 (same as the RQ1 dgemma job).
#    sinfo -o "%N %G" | grep -iE "6000|pro"     # find the GRES type
# Prereq (one-time): bash server/setup_dgemma_pylibs.sh
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch --gres=gpu:<type>:1 server/jobs/DiffStep-dgemma.slurm --max-samples 20 --debug   # smoke + dumps generation_config
#   sbatch --gres=gpu:<type>:1 server/jobs/DiffStep-dgemma.slurm                              # default (100 samples, grid<=48)
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs

DGEMMA_LIBS="$UMBRELLA/pylibs_dgemma"
if [ ! -d "$DGEMMA_LIBS/transformers" ]; then
  echo "FATAL: $DGEMMA_LIBS missing transformers (run: bash server/setup_dgemma_pylibs.sh)" >&2
  exit 1
fi
export PYTHONPATH="$DGEMMA_LIBS${PYTHONPATH:+:$PYTHONPATH}"

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${VRAM_MB:-0}" -lt 90000 ]; then
  echo "FATAL: GPU only ${VRAM_MB:-?}MiB; need a 96GB card. sinfo -o \"%N %G\" | grep -iE \"6000|pro\"" >&2
  exit 1
fi

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} DiffusionGemma step sweep ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'@',transformers.__file__)"
python -c "import importlib; importlib.import_module('transformers.models.diffusion_gemma'); print('diffusion_gemma OK')"

# NOTE: first run with --debug dumps model.generation_config so the real field
# names are visible; if auto-detect misses, re-run with
#   --max-steps-field <name> --early-stop-fields <bool1,bool2>
srun python -u experiments/exp_diffusion_steps_dgemma.py "$@"

echo "=== summary ==="
ls -lt results/diffusion_steps_benchmark/summary_dgemma_*.csv 2>/dev/null | head -2
