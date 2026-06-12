#!/bin/bash
#SBATCH --job-name=rid_DiffusionGemma-26B-A4B
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/DiffusionGemma-26B-A4B-%j.out
#SBATCH --error=logs/DiffusionGemma-26B-A4B-%j.err
# DiffusionGemma-26B-A4B refineID 推理 + 全site一致性门控指标。
#
# !! bf16 权重 ~52GB：A40(48GB) 必 OOM，必须落在 96GB RTX PRO 6000 上。
#    先查 GRES 型名：  sinfo -o "%N %G" | grep -i 6000
#    再带型名提交：    sbatch --gres=gpu:<型名>:1 server/jobs/DiffusionGemma-26B-A4B.slurm
#
# 前置（一次性）：bash server/setup_dgemma_pylibs.sh
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch --gres=gpu:<型名>:1 <this-file>                    # 全量
#   sbatch --gres=gpu:<型名>:1 <this-file> --max-samples 5 --debug   # 冒烟
set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh
mkdir -p logs
MODEL="DiffusionGemma-26B-A4B"

# 新 transformers 树 PREPEND（先于 Dream/DreamOn 钉死的 4.57.1 pylibs）。
DGEMMA_LIBS="$UMBRELLA/pylibs_dgemma"
if [ ! -d "$DGEMMA_LIBS" ]; then
  echo "FATAL: $DGEMMA_LIBS 不存在，先跑 bash server/setup_dgemma_pylibs.sh" >&2
  exit 1
fi
export PYTHONPATH="$DGEMMA_LIBS${PYTHONPATH:+:$PYTHONPATH}"

# M3 质量指标要词典；没有就让 runner 回退内置缩写表（不报错，质量分偏低）。
DICT_ARG=""
for d in /usr/share/dict/words "$UMBRELLA/words.txt"; do
  [ -f "$d" ] && DICT_ARG="--dict $d" && break
done

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} model:${MODEL} ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"
# 架构可用性探针：失败说明 PYTHONPATH 没吃到新 transformers
python -c "import importlib; importlib.import_module('transformers.models.diffusion_gemma'); print('diffusion_gemma OK')"

srun python experiments/run_all_refineID_unified.py --only "${MODEL}" ${DICT_ARG} "$@"

echo "=== results/unified_refineID/ ==="
ls -lt results/unified_refineID/predictions results/unified_refineID/metrics 2>/dev/null | head
