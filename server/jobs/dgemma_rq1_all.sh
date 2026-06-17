#!/bin/bash
# DiffusionGemma-26B-A4B 的 RQ1 一键编排（DAIC login 节点跑这个脚本本身，不要 sbatch 它）。
# 用 SLURM afterok 依赖把四步串起来，全程无人值守：
#
#   1) 冒烟 (DiffusionGemma-smoke.slurm)  -- 门控：预测全空则 exit!=0，后续自动不跑
#   2) 全量 RQ1 (DiffusionGemma-26B-A4B.slurm)  -- afterok 1)
#   3) LLM-as-Judge ×5（主环境 4.57.1，需 HF_TOKEN 给 gated Gemma-2-27B）  -- afterok 2)
#   4) 重建 metrics+leaderboard（--skip-inference, 纯 CPU）               -- afterok 3)
#
# 用法：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/setup_dgemma_pylibs.sh                 # 一次性（PREDOWNLOAD=1 顺便拉权重）
#   HF_TOKEN=hf_xxx bash server/jobs/dgemma_rq1_all.sh
#
# 可选环境变量：
#   GPU_TYPE=nvidia_rtx_pro_6000   # 不设则自动探测 96GB 卡的 GRES 名
#   SKIP_SMOKE=1                   # 已验证过冒烟，直接全量起步
#   SKIP_LJ=1                      # 只要预测，不跑 judge/rescore（此时不需要 HF_TOKEN）
set -euo pipefail
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
source server/env_daic.sh
mkdir -p logs

# --- 96GB 卡 GRES 名：优先用 GPU_TYPE，否则自动探测 ----------------------------
if [ -z "${GPU_TYPE:-}" ]; then
  GPU_TYPE=$(sinfo -h -o "%G" | tr ',' '\n' | grep -ioE "gpu:[^:]*6000[^:]*" | head -1 | cut -d: -f2 || true)
  if [ -n "$GPU_TYPE" ]; then
    echo "[auto] GPU_TYPE=$GPU_TYPE"
  else
    echo "FATAL: 没探测到 96GB 卡。手动设 GPU_TYPE=…，候选见下：" >&2
    sinfo -o "%N %G" | grep -iE "6000|pro" >&2 || true
    exit 1
  fi
fi

# --- pylibs_dgemma 必须先装好（transformers 5.11 含 diffusion_gemma）-----------
DGLIBS="$UMBRELLA/pylibs_dgemma"
if [ ! -d "$DGLIBS/transformers" ]; then
  echo "FATAL: $DGLIBS 缺 transformers。先跑：bash server/setup_dgemma_pylibs.sh" >&2
  exit 1
fi

SKIP_LJ="${SKIP_LJ:-0}"
if [ "$SKIP_LJ" != "1" ]; then
  : "${HF_TOKEN:?跑 judge 需要 HF_TOKEN（gated Gemma-2-27B-It）。只想要预测就加 SKIP_LJ=1}"
  export HF_TOKEN
fi

BIG="--gres=gpu:${GPU_TYPE}:1 --mem=96G"
ALLJ="--judge-model Qwen2.5-7B-Instruct --judge-model Qwen2.5-14B-Instruct \
--judge-model Qwen2.5-32B-Instruct --judge-model Mistral-Small-24B --judge-model Gemma-2-27B-It"
MODEL="DiffusionGemma-26B-A4B"

echo "=== DiffusionGemma RQ1 编排  GPU=$GPU_TYPE  SKIP_SMOKE=${SKIP_SMOKE:-0}  SKIP_LJ=$SKIP_LJ ==="

# --- 1) 冒烟门控 ---------------------------------------------------------------
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  SID=$(sbatch --parsable $BIG --time=02:00:00 server/jobs/DiffusionGemma-smoke.slurm)
  echo "[1] smoke    = $SID  (logs/DiffusionGemma-smoke-$SID.out;门控：全空则失败、链中止)"
  DEP_FULL="--dependency=afterok:$SID"
else
  DEP_FULL=""
  echo "[1] smoke    = SKIPPED"
fi

# --- 2) 全量 RQ1 推理 ----------------------------------------------------------
FID=$(sbatch --parsable $DEP_FULL $BIG --time=24:00:00 server/jobs/DiffusionGemma-26B-A4B.slurm)
echo "[2] full     = $FID  (logs/DiffusionGemma-26B-A4B-$FID.out)"

if [ "$SKIP_LJ" = "1" ]; then
  echo "[3/4] LJ + rescore = SKIPPED (SKIP_LJ=1)。预测好后手动跑 judge / --skip-inference。"
  echo; squeue -u "$USER"; exit 0
fi

# --- 3) LLM-as-Judge ×5（主环境，不 PREPEND pylibs_dgemma）---------------------
LID=$(sbatch --parsable --dependency=afterok:$FID $BIG --time=20:00:00 \
      --job-name="lj-$MODEL" --output="logs/lj-$MODEL-%j.out" --error="logs/lj-$MODEL-%j.err" \
      --wrap="source server/env_daic.sh && export HF_TOKEN=${HF_TOKEN} && \
              python experiments/run_llm_judge_unified.py --only $MODEL $ALLJ")
echo "[3] judge×5  = $LID  (afterok full)"

# --- 4) 重建 metrics + leaderboard（CPU）--------------------------------------
RID=$(sbatch --parsable --dependency=afterok:$LID --cpus-per-task=4 --mem=16G --time=02:00:00 \
      --job-name="rescore-$MODEL" --output="logs/rescore-$MODEL-%j.out" --error="logs/rescore-$MODEL-%j.err" \
      --wrap="source server/env_daic.sh && python experiments/run_all_refineID_unified.py --skip-inference")
echo "[4] rescore  = $RID  (afterok judge)"

echo; squeue -u "$USER"
echo
echo ">> 全链结束后（squeue 清空）合并 judge 矩阵："
echo "   source server/env_daic.sh && python experiments/run_llm_judge_unified.py --combine-only"
echo ">> 再把这几个 CSV 拉回 Mac 出图（见 docs/diffusiongemma_runbook.md 第 5-6 步）："
echo "   results/unified_refineID/predictions/$MODEL.csv"
echo "   results/unified_refineID/llm_judge/${MODEL}__judge_*.csv"
echo "   results/unified_refineID/leaderboard.csv"
