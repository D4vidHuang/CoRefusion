#!/bin/bash
#SBATCH --job-name=rid_DreamCoder-7B
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/DreamCoder-7B-%j.out
#SBATCH --error=logs/DreamCoder-7B-%j.err
# 单模型 refineID 推理 + 全site一致性门控指标；结果统一写 results/unified_refineID/。
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch <this-file>                 # 全量
#   sbatch <this-file> --max-samples 20  # 冒烟（脚本名后的参数原样透传给 runner）
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs
MODEL="DreamCoder-7B"

# M3 质量指标要词典；没有就让 runner 回退内置缩写表（不报错，质量分偏低）。
DICT_ARG=""
for d in /usr/share/dict/words "$UMBRELLA/words.txt"; do
  [ -f "$d" ] && DICT_ARG="--dict $d" && break
done

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} model:${MODEL} ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

srun python experiments/run_all_refineID_unified.py --only "${MODEL}" ${DICT_ARG} "$@"

echo "=== results/unified_refineID/ ==="
ls -lt results/unified_refineID/predictions results/unified_refineID/metrics 2>/dev/null | head
