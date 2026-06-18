#!/bin/bash
# 在 DAIC 上不方便用 git 时，用 raw.githubusercontent 把 “LJ 判 RefineID 自身
# ground truth”（judge ceiling）实验所需文件拉到位。
#
# 用法（DAIC，任意目录直接跑）：
#   curl -fsSL https://raw.githubusercontent.com/D4vidHuang/CoRefusion/main/server/jobs/fetch_refineid_gt_judge.sh | bash
#
# 之后（在仓库根）：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch server/run_llm_judge_groundtruth.slurm --max-samples 50   # 先冒烟
#   sbatch server/run_llm_judge_groundtruth.slurm                    # 全量（默认 7B judge）
set -euo pipefail
PROJ="/tudelft.net/staff-umbrella/CoReFusion/CoRefusion"
REF="${REF:-main}"                                  # 可用 REF=xxx 覆盖分支/commit
RAW="https://raw.githubusercontent.com/D4vidHuang/CoRefusion/$REF"
cd "$PROJ"
mkdir -p experiments server/jobs logs

# 拉取列表：本实验新增的两个文件 + 它复用的 judge 主模块（未改也覆盖一遍，保证一致）。
FILES=(
  experiments/judge_refineid_ground_truth.py
  experiments/llm_judge_variable_naming.py
  server/run_llm_judge_groundtruth.slurm
  server/jobs/gt_judge_multi.sh
)
TS=$(date +%s)   # cache-buster：绕开 raw 的 ~5 分钟 CDN 缓存
for f in "${FILES[@]}"; do
  if curl -fsSL "$RAW/$f?t=$TS" -o "$f"; then
    echo "[ok]   $f ($(wc -l < "$f") 行)"
  else
    echo "[FAIL] $f" >&2; exit 1
  fi
done

# 校验关键文件确实到位（raw 缓存偶尔滞后）。
if grep -q "GROUND TRUTH" experiments/judge_refineid_ground_truth.py \
   && grep -q "mask-fill" experiments/judge_refineid_ground_truth.py; then
  echo "OK: ground-truth judge 脚本已就位（含 --mask-fill）"
else
  echo "WARN: 脚本疑似旧版（raw CDN 缓存？等几分钟重跑本脚本）" >&2
fi

# 数据集 data/test.csv 体积大且服务器上已有，不走 raw；缺失才提示。
if [ ! -f data/test.csv ]; then
  echo "WARN: 缺 data/test.csv（1000 条 RefineID 基准），请确认它在仓库里。" >&2
fi

chmod +x server/jobs/gt_judge_multi.sh 2>/dev/null || true

echo
echo "下一步："
echo "  cd $PROJ && mkdir -p logs"
echo "  # 单 judge（默认 Qwen2.5-7B）："
echo "  sbatch server/run_llm_judge_groundtruth.slurm --max-samples 50   # 冒烟"
echo "  sbatch server/run_llm_judge_groundtruth.slurm                    # 全量"
echo "  # 复现全部 5-judge 面板（一卡一 judge 并行）："
echo "  GPU_TYPE=<GRES型名> HF_TOKEN=hf_xxx bash server/jobs/gt_judge_multi.sh"
echo "  # 全部结束后合并跨 judge 表："
echo "  python experiments/judge_refineid_ground_truth.py --combine-only"
