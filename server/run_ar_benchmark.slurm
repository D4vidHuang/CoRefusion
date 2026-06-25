#!/bin/bash

#SBATCH --job-name=corefusion_ar
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ar-%j.out
#SBATCH --error=logs/ar-%j.err
# 不写 --account / --partition(用 DAIC 默认,跟师兄一致)。gres 用 a40;若报 "Invalid GRES" 换成 gpu:nvidia_a40:1。
# qos 默认 normal;若 12h 被拒说明 normal 有更短时限,调小 --time。

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs

# 默认模型,可用 MODEL=... 覆盖。
MODEL="${MODEL:-Qwen2.5-Coder-7B}"

# M3 质量指标要词典;DAIC 计算节点不一定有 /usr/share/dict/words。
# 有则用,没有就让脚本回退到内置缩写表(质量分会偏低但不报错)。
DICT_ARG=""
for d in /usr/share/dict/words "$UMBRELLA/words.txt"; do
  if [ -f "$d" ]; then DICT_ARG="--dict $d"; break; fi
done

echo "=== node : $(hostname) | jobid: ${SLURM_JOB_ID:-NA} | model: ${MODEL} ==="
nvidia-smi || true
python -c "import torch, transformers; print('torch', torch.__version__, '| tfm', transformers.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU'))"

# AR 模型走 fim 引擎(标准 AutoModelForCausalLM),输出逐 site 预测后用一致性门控指标评估。
srun python experiments/run_all_refineID_unified.py --only "${MODEL}" ${DICT_ARG} "$@"

echo ""
echo "=== outputs (results/unified_refineID/) ==="
ls -lt results/unified_refineID/predictions results/unified_refineID/metrics 2>/dev/null | head -20
echo "--- leaderboard ---"
cat results/unified_refineID/leaderboard.csv 2>/dev/null || echo "(no leaderboard yet)"
