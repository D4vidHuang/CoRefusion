#!/bin/bash
# ============================================================================
# Fig 6 (diffusion-step sensitivity) -- FULL 1000 RefineID samples, all 3 dLLMs.
# Run ON DAIC from the repo root (VPN on):
#     cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#     git pull
#     bash server/jobs/submit_fig6_full.sh
#
# What it submits:
#   * DiffuCoder-7B  : 1 job, full 1000, grid 1 2 4 8 16 32 64  (A40 ok)
#   * DreamCoder-7B  : 1 job, full 1000, grid 1 2 4 8 16 32 64  (A40 ok)
#       The T=64 pass alone is ~5h; the whole sweep is ~11-14h, inside the 24h wall.
#   * DiffusionGemma : ONE JOB PER STEP value (1 2 4 8 16 32 48 64), full 1000.
#       Forcing all denoising steps on 1000 samples is ~8x its RQ1 cost, so a single
#       combined job would time out -- per-step jobs each get their own 24h budget,
#       run in parallel, and fig07 merges the per-step summary CSVs. Needs the 96GB
#       card (RTX PRO 6000); the dgemma script must be the version with --sample
#       (git pull / checkout origin/main if `--help` doesn't list it).
#
# Knobs (env vars):
#   DGEMMA_GRES=gpu:<type>:1   GRES for the dgemma jobs (default nvidia_rtx_pro_6000)
#   DGEMMA_STEPS="1 2 4 ..."   step grid for dgemma         (default 1 2 4 8 16 32 48 64)
#   SKIP_DGEMMA=1              submit only DiffuCoder/DreamCoder
#   SKIP_FIXED=1              submit only DiffusionGemma
# Results -> results/diffusion_steps_benchmark/{summary_*.csv, summary_dgemma_*.csv}
# Pull back with the rsync in pull_dreamon_results.sh's diffusion_steps section.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

DGEMMA_GRES="${DGEMMA_GRES:-gpu:nvidia_rtx_pro_6000:1}"
DGEMMA_STEPS="${DGEMMA_STEPS:-1 2 4 8 16 32 48 64}"

echo "repo: $REPO"

if [ "${SKIP_FIXED:-0}" != "1" ]; then
  echo "== DiffuCoder-7B  (full 1000, grid 1 2 4 8 16 32 64) =="
  MODEL=DiffuCoder-7B sbatch server/jobs/DiffStep-sweep.slurm
  echo "== DreamCoder-7B  (full 1000, grid 1 2 4 8 16 32 64) =="
  MODEL=DreamCoder-7B sbatch server/jobs/DiffStep-sweep.slurm
fi

if [ "${SKIP_DGEMMA:-0}" != "1" ]; then
  echo "== DiffusionGemma-26B-A4B  (full 1000, one job per step: $DGEMMA_STEPS) =="
  for T in $DGEMMA_STEPS; do
    sbatch --gres="$DGEMMA_GRES" server/jobs/DiffStep-dgemma.slurm --sample 0 --steps "$T"
  done
fi

echo
echo "submitted. watch:   squeue -u \$USER"
echo "results dir:        results/diffusion_steps_benchmark/"
