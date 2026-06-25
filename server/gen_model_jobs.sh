#!/bin/bash
# 为 registry 里每个模型生成一个独立的 slurm 脚本到 server/jobs/。
# 每个脚本可单独 sbatch；结果都写到同一个文件夹 results/unified_refineID/。
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/gen_model_jobs.sh                       # 生成 server/jobs/*.slurm
#   mkdir -p logs
#   sbatch server/jobs/Qwen2.5-Coder-7B.slurm           # 单独跑一个（全量）
#   sbatch server/jobs/Qwen2.5-Coder-7B.slurm --max-samples 20   # 冒烟
#   bash   server/jobs/aggregate.sh                      # 全跑完后合并 leaderboard
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="$HERE/jobs"
mkdir -p "$JOBS"

# name | a40_gpus | host_mem | walltime | gated(1=需要 HF_TOKEN)
# 单卡 A40(48GB) 足够所有模型：15B bf16 ~30GB、16B fp16 ~32GB 都放得下。
MODELS="
DreamCoder-7B|1|48G|16:00:00|0
DiffuCoder-7B|1|48G|16:00:00|0
DreamOn-7B|1|48G|16:00:00|0
CodeLlama-13B|1|64G|18:00:00|1
CodeLlama-7B|1|48G|12:00:00|1
StarCoder2-15B|1|64G|18:00:00|0
StarCoder2-7B|1|48G|12:00:00|0
StarCoder2-3B|1|32G|08:00:00|0
DeepSeek-Coder-6.7B|1|48G|12:00:00|0
DeepSeek-Coder-1.3B|1|32G|06:00:00|0
Qwen2.5-Coder-14B|1|64G|18:00:00|0
Qwen2.5-Coder-7B|1|48G|12:00:00|0
Qwen2.5-Coder-3B|1|32G|08:00:00|0
Qwen2.5-Coder-1.5B|1|32G|06:00:00|0
CodeGemma-7B|1|48G|12:00:00|1
CodeGemma-2B|1|32G|08:00:00|1
CodeT5p-16B|1|96G|30:00:00|0
CodeT5p-6B|1|48G|16:00:00|0
CodeT5p-2B|1|32G|08:00:00|0
CodeT5-large|1|24G|04:00:00|0
CodeT5-base|1|24G|04:00:00|0
CodeT5-small|1|24G|04:00:00|0
"

emit() {
  local name="$1" gpus="$2" mem="$3" time="$4" gated="$5"
  local f="$JOBS/${name}.slurm"

  cat > "$f" <<EOF
#!/bin/bash
#SBATCH --job-name=rid_${name}
#SBATCH --gres=gpu:${gpus}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=logs/${name}-%j.out
#SBATCH --error=logs/${name}-%j.err
EOF

  if [ "$gated" = "1" ]; then
    printf '%s\n' "# ⚠ gated 模型：先 export HF_TOKEN=hf_xxx 再 sbatch（或 token 已在 hf_cache 里）。" >> "$f"
  fi

  cat >> "$f" <<'EOF'
# 单模型 refineID 推理 + 全site一致性门控指标；结果统一写 results/unified_refineID/。
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch <this-file>                 # 全量
#   sbatch <this-file> --max-samples 20  # 冒烟（脚本名后的参数原样透传给 runner）
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
mkdir -p logs
EOF

  printf 'MODEL=%s\n' "\"${name}\"" >> "$f"

  cat >> "$f" <<'EOF'

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
EOF

  echo "  wrote jobs/${name}.slurm   (gpu:${gpus}, ${mem}, ${time}$([ "$gated" = 1 ] && echo ', GATED'))"
}

echo "Generating per-model slurm scripts into $JOBS ..."
while IFS='|' read -r name gpus mem time gated; do
  [ -z "${name// }" ] && continue
  emit "$name" "$gpus" "$mem" "$time" "$gated"
done <<< "$MODELS"

# 合并 leaderboard 的小工具（纯 CPU，login 节点直接 bash 跑，不用 sbatch）。
cat > "$JOBS/aggregate.sh" <<'EOF'
#!/bin/bash
# 所有单模型 job 跑完后，重新评分 predictions/ 下所有已存在的模型，
# 生成合并后的 results/unified_refineID/leaderboard.csv。纯 CPU。
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
DICT_ARG=""
for d in /usr/share/dict/words "$UMBRELLA/words.txt"; do
  [ -f "$d" ] && DICT_ARG="--dict $d" && break
done
python experiments/run_all_refineID_unified.py --skip-inference $DICT_ARG
echo "---- leaderboard (results/unified_refineID/leaderboard.csv) ----"
cat results/unified_refineID/leaderboard.csv
EOF
chmod +x "$JOBS"/aggregate.sh 2>/dev/null || true

echo ""
echo "Done -> $JOBS"
echo "提交全部：for f in server/jobs/*.slurm; do sbatch \"\$f\"; done"
