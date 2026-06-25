#!/bin/bash
# Smoke test: DreamOn-0.5B-Java on the first 5 samples. 验证整条链路跑通。
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   sbatch server/smoke_test.slurm
#   squeue -u $USER ; tail -f logs/smoke_<jobid>.out
#
#SBATCH --job-name=corefusion_smoke
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
# 跟师兄验证过的脚本一致:不写 --account / --partition,用 DAIC 默认(默认分区=all)。
# gres 用 a40(他用的 gpu:a40:3)。若报 "Invalid GRES",换成 gpu:nvidia_a40:1。

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"

echo "=== node : $(hostname) | jobid: ${SLURM_JOB_ID:-NA} ==="
nvidia-smi || true
python -c "import torch, transformers; print('torch', torch.__version__, '| tfm', transformers.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU'))"

srun python experiments/benchmark_dreamon_java_identifiers.py \
    --model dreamon-0.5b-Java \
    --max-samples 5 \
    --debug

echo ""
echo "=== results in results/dreamon_java/ ==="
ls -lt results/dreamon_java/ | head
