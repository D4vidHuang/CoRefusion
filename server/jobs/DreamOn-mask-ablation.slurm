#!/bin/bash
#SBATCH --job-name=dreamon_mask_abl
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dreamon-mask-abl-%j.out
#SBATCH --error=logs/dreamon-mask-abl-%j.err
# DreamOn-7B mask-token count ablation (k=1..5) -> Fig 5 (fig06_mask_count_ablation).
# Variable-canvas dLLM: k is the INITIAL canvas. Uses the main env (transformers
# 4.57.1, Dream/DreamOn custom code) -- NOT the dgemma tree.
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   # RECOMMENDED: 5 parallel single-k jobs, each on the SAME random 100 samples
#   # (fixed seed) with a live tqdm bar; each writes its own result file on finish:
#   for k in 1 2 3 4 5; do sbatch server/jobs/DreamOn-mask-ablation.slurm --mask-counts $k; done
#   # all k in one job:           sbatch server/jobs/DreamOn-mask-ablation.slurm
#   # full 1000 samples instead:  sbatch server/jobs/DreamOn-mask-ablation.slurm --mask-counts 2 --sample 0
#
# Default = random 100 samples (--sample 100 --seed 42; identical set across all k).
# Output: results/1t5t_exp/dreamon_mask_ablation_k<k>tok_<ts>.csv (one per k; read by fig06).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} DreamOn mask ablation ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

srun python -u experiments/1t5t_exp/part2_dreamon_mask_ablation.py "$@"

echo "=== summary (results/1t5t_exp/dreamon_mask_ablation_summary.csv) ==="
cat results/1t5t_exp/dreamon_mask_ablation_summary.csv 2>/dev/null || true
