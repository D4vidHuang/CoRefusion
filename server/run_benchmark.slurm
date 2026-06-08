#!/bin/bash
# Full EM benchmark on data/test.csv. 默认跑两个模型(0.5B + 7B)全量样本。
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   sbatch server/run_benchmark.slurm                          # 两个模型,全量
#   sbatch server/run_benchmark.slurm --model dreamon-7b-Java  # 只跑 7B
#   sbatch server/run_benchmark.slurm --max-samples 100        # 限量
#   sbatch server/run_benchmark.slurm --hf-repo D4vidHuang/benchmark_ReFineID_DreamOn_Java
# (脚本名后面的参数都会原样传给 benchmark 脚本)
#
#SBATCH --job-name=corefusion_bench
#SBATCH --account=testusers
#SBATCH --partition=ewi-st
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh

echo "=== node : $(hostname) | jobid: ${SLURM_JOB_ID:-NA} ==="
nvidia-smi || true
python -c "import torch, transformers; print('torch', torch.__version__, '| tfm', transformers.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU'))"

# 脚本默认两个模型;命令行可用 --model / --max-samples / --hf-repo 等覆盖
srun python experiments/benchmark_dreamon_java_identifiers.py "$@"

echo ""
echo "=== results in results/dreamon_java/ ==="
ls -lt results/dreamon_java/ | head
