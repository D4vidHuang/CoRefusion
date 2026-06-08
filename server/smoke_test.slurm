#!/bin/bash
# Smoke test: DreamOn-0.5B-Java on the first 5 samples. 验证整条链路跑通。
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   sbatch server/smoke_test.slurm
#   squeue -u $USER ; tail -f logs/smoke_<jobid>.out
#
#SBATCH --job-name=corefusion_smoke
#SBATCH --account=testusers
#SBATCH --partition=ewi-st
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
# 想试 Blackwell 卡(需 torch>=2.7,module 的 2.5.1 大概率不支持)就换成:
##SBATCH --gres=gpu:nvidia_rtx_pro_6000:1

set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh

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
