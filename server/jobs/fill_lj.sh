#!/bin/bash
# 自动补齐 prediction + LLM-as-Judge,LJ 用 SLURM 依赖在 prediction 跑完后自动开始,
# 无需手动 sbatch。先在 Mac 上把两个 colab 预测传到 predictions/(覆盖截断版),再跑本脚本。
#
# 用法（DAIC login 节点）：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   GPU_TYPE=<96GB卡GRES名>  HF_TOKEN=hf_xxx  bash server/jobs/fill_lj.sh
#   #  GPU_TYPE: sinfo -o "%N %G" | grep -i 6000   （24B/27B/32B judge 放不进 48GB A40）
#   #  HF_TOKEN: Gemma-2-27B-It 是 gated 模型必须有
#
# 跑完后（squeue 清空）合并矩阵：
#   source server/env_daic.sh && python experiments/run_llm_judge_unified.py --combine-only
set -euo pipefail
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
mkdir -p logs
: "${GPU_TYPE:?set GPU_TYPE to the 96GB card GRES name (sinfo -o '%N %G' | grep -i 6000)}"
: "${HF_TOKEN:?set HF_TOKEN (Gemma-2-27B-It is gated)}"

ALLJ="--judge-model Qwen2.5-7B-Instruct --judge-model Qwen2.5-14B-Instruct \
--judge-model Qwen2.5-32B-Instruct --judge-model Mistral-Small-24B --judge-model Gemma-2-27B-It"
BIG="--gres=gpu:${GPU_TYPE}:1 --mem=96G"
ENV="source server/env_daic.sh && export HF_TOKEN=${HF_TOKEN}"

# ---- 0) DreamOn 预测去重（1142 -> 唯一 id，保留首次出现）-----------------------
python3 - <<'PY'
import csv; csv.field_size_limit(2**31-1)
p="results/unified_refineID/predictions/DreamOn-7B.csv"
rows=list(csv.reader(open(p))); seen=set(); out=[rows[0]]
for r in rows[1:]:
    if r and r[0] not in seen: seen.add(r[0]); out.append(r)
csv.writer(open(p,"w",newline="")).writerows(out)
print(f"[dedup] DreamOn-7B predictions -> {len(out)-1} rows")
PY

BENCH_IDS=""

# ---- 1) 截断预测：重跑 benchmark -> 跑完自动全判（afterok 依赖）---------------
for M in CodeT5p-2B CodeT5p-6B; do
  BID=$(sbatch --parsable --gres=gpu:1 --mem=64G --time=16:00:00 \
        --job-name="bench-$M" --output="logs/bench-$M-%j.out" --error="logs/bench-$M-%j.err" \
        --wrap="LOCAL_HF_CACHE=1 $ENV && python experiments/run_all_refineID_unified.py --only $M")
  BENCH_IDS="${BENCH_IDS:+$BENCH_IDS:}$BID"
  JID=$(sbatch --parsable --dependency=afterok:$BID $BIG --time=20:00:00 \
        --job-name="lj-$M" --output="logs/lj-$M-%j.out" --error="logs/lj-$M-%j.err" \
        --wrap="LOCAL_HF_CACHE=1 $ENV && python experiments/run_llm_judge_unified.py --only $M $ALLJ")
  echo "[chain] $M : bench=$BID -> LJ(all 5)=$JID (auto after bench)"
done

# ---- 2) Colab 已传的 DiffuCoder/DreamCoder：直接全判（预测已就位，无需 bench）---
JID=$(sbatch --parsable $BIG --time=20:00:00 \
      --job-name="lj-colab" --output="logs/lj-colab-%j.out" --error="logs/lj-colab-%j.err" \
      --wrap="LOCAL_HF_CACHE=1 $ENV && python experiments/run_llm_judge_unified.py \
              --only DiffuCoder-7B --only DreamCoder-7B $ALLJ")
echo "[colab] DiffuCoder+DreamCoder LJ(all 5) = $JID"

# ---- 3) 预测已完整、仅缺 Gemma 的模型：一个 Gemma job 用 --resume 补齐 ---------
GEMMA_ONLY="CodeT5p-16B DeepSeek-Coder-1.3B DeepSeek-Coder-6.7B Qwen2.5-Coder-1.5B \
Qwen2.5-Coder-3B Qwen2.5-Coder-7B Qwen2.5-Coder-14B StarCoder2-3B StarCoder2-7B \
StarCoder2-15B DreamOn-7B"
ONLY=""; for m in $GEMMA_ONLY; do ONLY="$ONLY --only $m"; done
JID=$(sbatch --parsable $BIG --time=18:00:00 \
      --job-name="lj-gemma" --output="logs/lj-gemma-%j.out" --error="logs/lj-gemma-%j.err" \
      --wrap="LOCAL_HF_CACHE=1 $ENV && python experiments/run_llm_judge_unified.py \
              --judge-model Gemma-2-27B-It --resume $ONLY")
echo "[gemma] Gemma-2-27B-It fill (11 models, --resume) = $JID"

# ---- 4) 全部预测就绪后，重建 metrics + leaderboard（纯 CPU，依赖两个 bench）----
if [ -n "$BENCH_IDS" ]; then DEP="--dependency=afterok:$BENCH_IDS"; else DEP=""; fi
RID=$(sbatch --parsable $DEP --cpus-per-task=4 --mem=16G --time=02:00:00 \
      --job-name="rescore" --output="logs/rescore-%j.out" --error="logs/rescore-%j.err" \
      --wrap="$ENV && python experiments/run_all_refineID_unified.py --skip-inference")
echo "[rescore] metrics+leaderboard rebuild = $RID (after benches)"

echo; squeue -u "$USER"
echo
echo ">> DiffusionGemma-26B-A4B 没纳入：它预测全空=模型本身没跑通，需单独 debug。"
echo ">> 全部结束后跑：python experiments/run_llm_judge_unified.py --combine-only"
