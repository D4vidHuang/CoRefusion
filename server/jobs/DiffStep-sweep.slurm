#!/bin/bash
#SBATCH --job-name=diffstep
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/diffstep-%j.out
#SBATCH --error=logs/diffstep-%j.err
# Diffusion-step sensitivity sweep (Fig 6) for a FIXED-CANVAS dLLM.
# Sweeps steps in model.diffusion_generate(steps=T) over T in {1,2,4,8,16,32,64}.
# Only DiffuCoder-7B and DreamCoder-7B expose this knob; DreamOn (variable canvas)
# and DiffusionGemma (block-AR) do NOT, and are intentionally excluded.
#
# Per model ~11-14h (the T=64 pass alone is ~5h on 1000 samples), so run the two
# models as SEPARATE parallel jobs:
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   MODEL=DreamCoder-7B sbatch server/jobs/DiffStep-sweep.slurm
#   MODEL=DiffuCoder-7B sbatch server/jobs/DiffStep-sweep.slurm   # re-run for same-GPU latency panel
#   # smoke (a few samples, all steps):
#   MODEL=DreamCoder-7B sbatch server/jobs/DiffStep-sweep.slurm --max-samples 20
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs
MODEL="${MODEL:-DreamCoder-7B}"

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} diffstep model:${MODEL} ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

srun python -u experiments/exp_diffusion_steps_benchmark.py \
     --model "${MODEL}" --steps 1 2 4 8 16 32 64 "$@"

echo "=== summary ==="
ls -lt results/diffusion_steps_benchmark/summary_*.csv 2>/dev/null | head -3
