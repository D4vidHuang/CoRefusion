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
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs
MODEL="DiffusionGemma-26B-A4B"

# 新 transformers 树 PREPEND（先于主环境钉死的 4.57.1）。
# 检查实际内容而非目录存在——setup 失败会留下空目录（269763 的教训）。
DGEMMA_LIBS="$UMBRELLA/pylibs_dgemma"
if [ ! -d "$DGEMMA_LIBS/transformers" ]; then
  echo "FATAL: $DGEMMA_LIBS 缺 transformers（目录空=上次 setup 失败）。" >&2
  echo "  先跑：bash server/setup_dgemma_pylibs.sh" >&2
  exit 1
fi
export PYTHONPATH="$DGEMMA_LIBS${PYTHONPATH:+:$PYTHONPATH}"

# VRAM 闸门：bf16 权重 ~52GB，46GB 的 L40/A40 必 OOM。在下载/加载权重之前
# 快速失败，而不是 20 分钟后 CUDA OOM。
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${VRAM_MB:-0}" -lt 90000 ]; then
  echo "FATAL: 本节点 GPU 只有 ${VRAM_MB:-?}MiB，需要 96GB 卡（RTX PRO 6000）。" >&2
  echo "  查型名：sinfo -o \"%N %G\" | grep -iE \"6000|pro\"" >&2
  echo "  再提交：sbatch --gres=gpu:<型名>:1 server/jobs/DiffusionGemma-26B-A4B.slurm" >&2
  exit 1
fi

# M3 质量指标要词典；没有就让 runner 回退内置缩写表（不报错，质量分偏低）。
DICT_ARG=""
for d in /usr/share/dict/words "$UMBRELLA/words.txt"; do
  [ -f "$d" ] && DICT_ARG="--dict $d" && break
done

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} model:${MODEL} ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'@',transformers.__file__,'| cuda',torch.cuda.is_available())"
# 架构可用性探针：失败说明 PYTHONPATH 没吃到新 transformers（需要 >=5.11.0）
python -c "import importlib; importlib.import_module('transformers.models.diffusion_gemma'); print('diffusion_gemma OK')"

srun python experiments/run_all_refineID_unified.py --only "${MODEL}" ${DICT_ARG} "$@"

echo "=== results/unified_refineID/ ==="
ls -lt results/unified_refineID/predictions results/unified_refineID/metrics 2>/dev/null | head
