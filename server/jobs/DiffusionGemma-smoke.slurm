#!/bin/bash
#SBATCH --job-name=rid_dgemma_smoke
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/DiffusionGemma-smoke-%j.out
#SBATCH --error=logs/DiffusionGemma-smoke-%j.err
# DiffusionGemma 冒烟 + 诊断：跑 standalone benchmark，--debug 打印 raw[0]/渲染后的
# prompt/out 类型/canvas 形状，并把每个 site 的诊断写成 JSONL，一把定位空输出原因。
#
# !! bf16 权重 ~52GB：A40(48GB) 必 OOM，必须落在 96GB RTX PRO 6000 上。
#    先查 GRES 型名：  sinfo -o "%N %G" | grep -iE "6000|pro"
#    再带型名提交：    sbatch --gres=gpu:<型名>:1 server/jobs/DiffusionGemma-smoke.slurm
#
# 前置（一次性）：bash server/setup_dgemma_pylibs.sh
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch --gres=gpu:<型名>:1 server/jobs/DiffusionGemma-smoke.slurm            # 默认 5 条
#   sbatch --gres=gpu:<型名>:1 server/jobs/DiffusionGemma-smoke.slurm --max-samples 10
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs

# 新 transformers 树 PREPEND（先于主环境钉死的 4.57.1）。检查实际内容而非目录存在。
DGEMMA_LIBS="$UMBRELLA/pylibs_dgemma"
if [ ! -d "$DGEMMA_LIBS/transformers" ]; then
  echo "FATAL: $DGEMMA_LIBS 缺 transformers（目录空=上次 setup 失败）。" >&2
  echo "  先跑：bash server/setup_dgemma_pylibs.sh" >&2
  exit 1
fi
export PYTHONPATH="$DGEMMA_LIBS${PYTHONPATH:+:$PYTHONPATH}"

# VRAM 闸门：bf16 权重 ~52GB，46GB 的 L40/A40 必 OOM。下载/加载前快速失败。
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${VRAM_MB:-0}" -lt 90000 ]; then
  echo "FATAL: 本节点 GPU 只有 ${VRAM_MB:-?}MiB，需要 96GB 卡（RTX PRO 6000）。" >&2
  echo "  查型名：sinfo -o \"%N %G\" | grep -iE \"6000|pro\"" >&2
  exit 1
fi

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} dgemma SMOKE ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'@',transformers.__file__,'| cuda',torch.cuda.is_available())"
python -c "import importlib; importlib.import_module('transformers.models.diffusion_gemma'); print('diffusion_gemma OK')"

# 默认 5 条；命令行可覆盖（追加在末尾，"$@" 透传给 python）。
ARGS=(--max-samples 5 --debug --debug-dump results/diffusiongemma_smoke/debug.jsonl)
# 不用 set -e 直接中断：先让 python 跑完（debug 已打印），再拿退出码当门控。
# python 在"全空"时 exit 2 -> 本 job 失败 -> afterok 链自动不跑全量（不浪费 24h）。
set +e
srun python experiments/benchmark_diffusiongemma.py "${ARGS[@]}" "$@"
RC=$?
set -e

echo "=== smoke output (exit=$RC) ==="
ls -l results/diffusiongemma_smoke/ 2>/dev/null || true
echo "--- raw[0] / preds 在上面日志里（搜 'dgemma debug'）；完整记录：results/diffusiongemma_smoke/debug.jsonl ---"
if [ "$RC" -ne 0 ]; then
  echo "!! 冒烟门控失败 (exit=$RC)：预测仍全空。看上面的 raw[0] 判断是生成还是抽取问题。" >&2
fi
exit $RC
