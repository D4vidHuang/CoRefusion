#!/bin/bash
#SBATCH --job-name=dreamon_k1k5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dreamon-k1k5-%j.out
#SBATCH --error=logs/dreamon-k1k5-%j.err
# ---------------------------------------------------------------------------
# DreamOn-7B mask-token ablation -- the TWO ENDPOINTS missing from Fig 5
# (fig06_mask_count_ablation): k=1 and k=5. FAST config for a same-day result:
# only 100 RefineID samples (fixed seed 42, so k=1 and k=5 see the SAME 100
# samples and are directly comparable). Both k finish well under the 6h budget
# (~20-40 min each on an A40); the 4h wall is generous head-room.
#
# QUICK START (on DAIC, with VPN, from the repo root):
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch server/jobs/DreamOn-mask-ablation-1and5.slurm
#
#   # FASTER (halves wall-clock): two parallel single-k jobs, race-free files
#   for k in 1 5; do sbatch server/jobs/DreamOn-mask-ablation-1and5.slurm --mask-counts $k; done
#
#   # need all five k instead:   sbatch server/jobs/DreamOn-mask-ablation-1and5.slurm --mask-counts 1 2 3 4 5
#   # full 1000 samples:         sbatch server/jobs/DreamOn-mask-ablation-1and5.slurm --sample 0
#
# Output -> results/1t5t_exp/dreamon_mask_ablation_k<k>tok_<ts>.csv  (read by fig06)
#        -> results/1t5t_exp/dreamon_mask_ablation_summary.csv
# Pull back to the Mac with:  bash pull_dreamon_results.sh
# ---------------------------------------------------------------------------
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs results/1t5t_exp

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} DreamOn k=1/k=5 ablation (n=100) ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

# Default: the two missing endpoints on 100 fixed samples. Any args passed to
# sbatch are forwarded verbatim and fully override the defaults (so e.g.
# `--mask-counts 1` or `--sample 0` behave exactly as written).
if [ "$#" -gt 0 ]; then
  ARGS=("$@")
else
  ARGS=(--mask-counts 1 5 --sample 100 --seed 42)
fi
echo "    python args: ${ARGS[*]}"

srun python -u experiments/1t5t_exp/part2_dreamon_mask_ablation.py "${ARGS[@]}"

echo "=== per-k result files ==="
ls -lt results/1t5t_exp/dreamon_mask_ablation_k*tok_*.csv 2>/dev/null | head
echo "=== summary (results/1t5t_exp/dreamon_mask_ablation_summary.csv) ==="
cat results/1t5t_exp/dreamon_mask_ablation_summary.csv 2>/dev/null || true
