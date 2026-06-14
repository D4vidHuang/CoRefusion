#!/bin/bash
#SBATCH --job-name=rq3_probe
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq3_probe-%j.out
#SBATCH --error=logs/rq3_probe-%j.err
# RQ3 v2 — GPU hidden-state extraction for the linear-probe redesign.
# Only the extraction needs the GPU; probe training + plotting are CPU and run
# afterwards (here best-effort, or on the login node).
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch server/jobs/rq3_extract.slurm                       # full 230 snippets
#   sbatch server/jobs/rq3_extract.slurm --max-snippets 8      # smoke test
#   sbatch server/jobs/rq3_extract.slurm --steps 64            # match RQ1/RQ2 gen config
#
# Args after the script name pass through to extract_probe_states.py.
#
# BEFORE SUBMIT, confirm two things (see docs/rq3_redesign.md "Open decisions"):
#   (1) --model matches the checkpoint RQ1/RQ2 EM/LJ were reported on (default
#       apple/DiffuCoder-7B-Base; the legacy experiment_a/b/c used -Instruct).
#   (2) --steps matches the RQ1/RQ2 generation step count (default 32 = paper RQ3;
#       the unified runner uses 64). DCL is step-count dependent.
set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh
mkdir -p logs results/rq3_probe figures/new/rq3

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

# ---- 1) GPU extraction (the only part that needs a GPU) --------------------
srun python experiments/rq3_probe/extract_probe_states.py \
     --model apple/DiffuCoder-7B-Base \
     --data data/test_filtered_1024.csv \
     --out-dir results/rq3_probe \
     --steps 32 --num-mask 2 "$@"

# ---- 2) probe training + figures (CPU; best-effort in-job) -----------------
# sklearn may not be in the uv venv; if not, run these two lines on the login
# node after:  uv pip install scikit-learn matplotlib
if python -c "import sklearn, matplotlib" 2>/dev/null; then
  python experiments/rq3_probe/train_probes.py --out-dir results/rq3_probe
  python experiments/rq3_probe/plot_rq3.py --out-dir results/rq3_probe --fig-dir figures/new/rq3
  echo "=== figures/new/rq3 ==="; ls -lt figures/new/rq3 2>/dev/null | head
else
  echo "[note] scikit-learn/matplotlib not in env — run on login node:"
  echo "       uv pip install scikit-learn matplotlib"
  echo "       python experiments/rq3_probe/train_probes.py --out-dir results/rq3_probe"
  echo "       python experiments/rq3_probe/plot_rq3.py --out-dir results/rq3_probe --fig-dir figures/new/rq3"
fi

echo "=== results/rq3_probe ==="; ls -lt results/rq3_probe 2>/dev/null | head
