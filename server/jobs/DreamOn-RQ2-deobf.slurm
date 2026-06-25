#!/bin/bash
#SBATCH --job-name=dreamon_rq2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dreamon-rq2-%j.out
#SBATCH --error=logs/dreamon-rq2-%j.err
# DreamOn-7B RQ2 deobfuscation: both all-masked AND target-only, one job.
# -> results/deobfuscation_refineID/DreamOn-7B_{all-masked,target-only}_<ts>.csv
# which Table tab:rq2_deobfuscation + Fig 8 pick up via
# analysis/reproduce_rq2_deobfuscation.py (DreamOn is auto-discovered there).
#
# Main env (transformers 4.57.1, DreamOn custom diffusion_generate) -- NOT dgemma.
# DiffusionGemma is block-AR (RQ1-only) and is intentionally NOT run here.
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch server/jobs/DreamOn-RQ2-deobf.slurm                       # both modes (all-masked then target-only)
#   sbatch server/jobs/DreamOn-RQ2-deobf.slurm --max-samples 20      # smoke, both modes
#   MODE=target-only sbatch server/jobs/DreamOn-RQ2-deobf.slurm      # ONE mode only (run the 2 modes
#   MODE=all-masked  sbatch server/jobs/DreamOn-RQ2-deobf.slurm      #   as parallel jobs; all-masked is slow)
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs

# MODE env: "all-masked" / "target-only" runs just that one; unset = both.
MODES="${MODE:-all-masked target-only}"

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} DreamOn RQ2 deobfuscation  modes=[$MODES] ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

for m in $MODES; do
  echo "--- mode: $m ---"
  srun python -u experiments/experiment_deobfuscation_refineID.py --model DreamOn-7B --mode "$m" "$@"
done

echo "=== DreamOn-7B deobfuscation results ==="
ls -lt results/deobfuscation_refineID/DreamOn-7B_* 2>/dev/null | head
