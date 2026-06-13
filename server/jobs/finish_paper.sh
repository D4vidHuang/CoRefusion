#!/bin/bash
# 一次性补齐论文所有更新。预测跑完用 SLURM 依赖自动接 LJ;3 套环境自动切换。
# 取代 fill_lj.sh（这个多做了 DiffusionGemma + DreamOn-Java + 0.5B + colab 上传）。
#
# 用法（DAIC login 节点）：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   GPU_TYPE=nvidia_rtx_pro_6000  HF_TOKEN=hf_xxx  bash server/jobs/finish_paper.sh
#   # 已经单独重跑了 7B-Java start0？  SKIP_DJAVA_START0=1 ... bash ...
#   # 不想跑 0.5B 全量？               SKIP_05B=1 ...
#
# 全部结束后（squeue 清空）：
#   source server/env_daic.sh && python experiments/run_llm_judge_unified.py --combine-only
set -euo pipefail
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
mkdir -p logs
: "${GPU_TYPE:?set GPU_TYPE=nvidia_rtx_pro_6000 (96GB card; sinfo -o '%N %G' | grep -i 6000)}"
: "${HF_TOKEN:?set HF_TOKEN (gated Gemma-2-27B judge + DiffusionGemma download)}"
export HF_TOKEN

RAW="https://raw.githubusercontent.com/D4vidHuang/CoRefusion/main"
PRED="results/unified_refineID/predictions"
DGLIBS="/tudelft.net/staff-umbrella/CoReFusion/pylibs_dgemma"   # tfm 5.11 (diffusion_gemma)
ALLJ="--judge-model Qwen2.5-7B-Instruct --judge-model Qwen2.5-14B-Instruct \
--judge-model Qwen2.5-32B-Instruct --judge-model Mistral-Small-24B --judge-model Gemma-2-27B-It"
BIG="--gres=gpu:${GPU_TYPE}:1 --mem=96G"
BENCH_IDS=""   # unified predictions that change -> rescore depends on these

# ===== 0) colab 全量预测覆盖截断版(git-raw,绕开 rsync 临时文件限制)=====
for M in DiffuCoder-7B DreamCoder-7B; do
  curl -fsSL "$RAW/transfer/$M.csv" -o "$PRED/$M.csv"
  echo "[pred] $M <- colab ($(($(wc -l < "$PRED/$M.csv")-1)) rows)"
done

# ===== 1) DreamOn-7B 预测去重(1142 -> 唯一 id)=====
python3 - <<'PY'
import csv; csv.field_size_limit(2**31-1)
p="results/unified_refineID/predictions/DreamOn-7B.csv"
rows=list(csv.reader(open(p))); seen=set(); out=[rows[0]]
for r in rows[1:]:
    if r and r[0] not in seen: seen.add(r[0]); out.append(r)
csv.writer(open(p,"w",newline="")).writerows(out)
print(f"[dedup] DreamOn-7B -> {len(out)-1} rows")
PY

# ===== 2) CodeT5p-2B/6B:重跑预测 -> afterok 自动全判 =====
for M in CodeT5p-2B CodeT5p-6B; do
  BID=$(sbatch --parsable --gres=gpu:1 --mem=64G --time=16:00:00 \
        --job-name="bench-$M" --output="logs/bench-$M-%j.out" --error="logs/bench-$M-%j.err" \
        --wrap="LOCAL_HF_CACHE=1 source server/env_daic.sh && python experiments/run_all_refineID_unified.py --only $M")
  BENCH_IDS="${BENCH_IDS:+$BENCH_IDS:}$BID"
  sbatch --parsable --dependency=afterok:$BID $BIG --time=20:00:00 \
        --job-name="lj-$M" --output="logs/lj-$M-%j.out" --error="logs/lj-$M-%j.err" \
        --wrap="LOCAL_HF_CACHE=1 source server/env_daic.sh && python experiments/run_llm_judge_unified.py --only $M $ALLJ"
  echo "[chain] $M bench=$BID -> LJ(all5) auto-after"
done

# ===== 3) DiffuCoder/DreamCoder(colab 已就位):全判 5 =====
sbatch --parsable $BIG --time=20:00:00 \
  --job-name="lj-colab" --output="logs/lj-colab-%j.out" --error="logs/lj-colab-%j.err" \
  --wrap="LOCAL_HF_CACHE=1 source server/env_daic.sh && python experiments/run_llm_judge_unified.py --only DiffuCoder-7B --only DreamCoder-7B $ALLJ" >/dev/null
echo "[lj] DiffuCoder+DreamCoder all-5"

# ===== 4) 预测已齐、仅缺 Gemma 的 11 个模型:一个 Gemma job --resume =====
GO="CodeT5p-16B DeepSeek-Coder-1.3B DeepSeek-Coder-6.7B Qwen2.5-Coder-1.5B Qwen2.5-Coder-3B \
Qwen2.5-Coder-7B Qwen2.5-Coder-14B StarCoder2-3B StarCoder2-7B StarCoder2-15B DreamOn-7B"
ONLY=""; for m in $GO; do ONLY="$ONLY --only $m"; done
sbatch --parsable $BIG --time=18:00:00 \
  --job-name="lj-gemma" --output="logs/lj-gemma-%j.out" --error="logs/lj-gemma-%j.err" \
  --wrap="LOCAL_HF_CACHE=1 source server/env_daic.sh && python experiments/run_llm_judge_unified.py --judge-model Gemma-2-27B-It --resume $ONLY" >/dev/null
echo "[lj] Gemma-2-27B fill (11 models, --resume)"

# ===== 5) DiffusionGemma:冒烟 -> afterok 全量 -> afterok 全判(pylibs_dgemma,无 LOCAL_HF_CACHE)=====
SID=$(sbatch --parsable $BIG --time=03:00:00 \
      --job-name="dg-smoke" --output="logs/dg-smoke-%j.out" --error="logs/dg-smoke-%j.err" \
      --wrap="source server/env_daic.sh && export PYTHONPATH=\"$DGLIBS:\${PYTHONPATH:-}\" && python experiments/benchmark_diffusiongemma.py --max-samples 3 --debug")
FID=$(sbatch --parsable --dependency=afterok:$SID $BIG --time=24:00:00 \
      --job-name="dg-full" --output="logs/dg-full-%j.out" --error="logs/dg-full-%j.err" \
      --wrap="source server/env_daic.sh && export PYTHONPATH=\"$DGLIBS:\${PYTHONPATH:-}\" && python experiments/run_all_refineID_unified.py --only DiffusionGemma-26B-A4B")
BENCH_IDS="${BENCH_IDS:+$BENCH_IDS:}$FID"
sbatch --parsable --dependency=afterok:$FID $BIG --time=20:00:00 \
      --job-name="lj-dgemma" --output="logs/lj-dgemma-%j.out" --error="logs/lj-dgemma-%j.err" \
      --wrap="source server/env_daic.sh && python experiments/run_llm_judge_unified.py --only DiffusionGemma-26B-A4B $ALLJ" >/dev/null
echo "[chain] DiffusionGemma smoke=$SID -> full=$FID -> LJ(all5) auto-after"
echo "        ^^ 看 logs/dg-smoke-*.out 的 raw[0];若仍空,scancel $FID 后贴我"

# ===== 6) DreamOn-7B-Java canvas start0 补跑(超时那块,拆 0-99 / 100-199)=====
if [ "${SKIP_DJAVA_START0:-0}" != "1" ]; then
  for S in 0 100; do
    sbatch --gres=gpu:1 --mem=64G --time=16:00:00 \
      --job-name="djava-s$S" --output="logs/djava-s$S-%j.out" --error="logs/djava-s$S-%j.err" \
      --wrap="source server/env_daic.sh && python experiments/benchmark_dreamon_java_identifiers.py --model dreamon-7b-Java --mask-style canvas --start $S --max-samples 100" >/dev/null
    echo "[djava] 7B canvas start=$S (100 samples)"
  done
fi

# ===== 7) (可选) DreamOn-0.5B-Java 全量 canvas(官方 origin 配方)=====
if [ "${SKIP_05B:-0}" != "1" ]; then
  sbatch --gres=gpu:1 --mem=48G --time=16:00:00 \
    --job-name="djava-05b" --output="logs/djava-05b-%j.out" --error="logs/djava-05b-%j.err" \
    --wrap="source server/env_daic.sh && python experiments/benchmark_dreamon_java_identifiers.py --model dreamon-0.5b-Java --mask-style canvas --alg origin --temperature 0 --steps 512" >/dev/null
  echo "[djava] 0.5B canvas full (origin/512/temp0)"
fi

# ===== 8) 全预测就绪后重建 metrics + leaderboard(纯 CPU,afterok benches)=====
DEP=""; [ -n "$BENCH_IDS" ] && DEP="--dependency=afterok:$BENCH_IDS"
sbatch $DEP --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --job-name="rescore" --output="logs/rescore-%j.out" --error="logs/rescore-%j.err" \
  --wrap="source server/env_daic.sh && python experiments/run_all_refineID_unified.py --skip-inference" >/dev/null
echo "[rescore] metrics+leaderboard rebuild after benches"

echo; squeue -u "$USER"
echo; echo ">> 全部结束后:source server/env_daic.sh && python experiments/run_llm_judge_unified.py --combine-only"
