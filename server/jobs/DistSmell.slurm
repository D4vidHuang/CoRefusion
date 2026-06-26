#!/bin/bash
#SBATCH --job-name=distsmell
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/distsmell-%j.out
#SBATCH --error=logs/distsmell-%j.err
# RQ3 distribution-based smell rank (revised Fig.10 / Table V).
# For each RefineID target position, scan the model's FULL output distribution
# and record the median rank of smell names the model itself surfaces (objective
# rule in smell_rule.py) vs the developer name, stratified by confidence regime.
# Replaces the old injected x/tmp/foo comparison. One forward pass per sample,
# so this is fast (~minutes-to-1h on 1000 samples), much cheaper than the step sweep.
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   MODEL=DiffuCoder-7B-Base sbatch server/jobs/DistSmell.slurm
#   MODEL=DreamCoder-7B      sbatch server/jobs/DistSmell.slurm
#   # smoke (a few samples):
#   MODEL=DiffuCoder-7B-Base sbatch server/jobs/DistSmell.slurm --max-samples 20
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs
MODEL="${MODEL:-DiffuCoder-7B-Base}"      # keys: DiffuCoder-7B-Base / DiffuCoder-7B-Instruct / DreamCoder-7B

# The smell rule exempts real short words via a dictionary; reuse the DAIC
# convention (the rule also finds these internally, but pass it explicitly).
DICT_ARG=""
for d in /usr/share/dict/words "$UMBRELLA/words.txt"; do
  [ -f "$d" ] && DICT_ARG="--dict $d" && break
done

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} distsmell model:${MODEL} ${DICT_ARG} ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

# -u => unbuffered, so the tqdm progress bars (vocab scan + per-sample loop) stream live to the log.
srun python -u experiments/exp_distribution_smell_rank.py \
     --model "${MODEL}" ${DICT_ARG} "$@"

echo "=== outputs (results/distribution_smell_rank/) ==="
ls -lt results/distribution_smell_rank/${MODEL}_*_summary.csv 2>/dev/null | head -3
echo "--- summary ---"
tail -n +1 "$(ls -t results/distribution_smell_rank/${MODEL}_*_summary.csv 2>/dev/null | head -1)" 2>/dev/null || true
