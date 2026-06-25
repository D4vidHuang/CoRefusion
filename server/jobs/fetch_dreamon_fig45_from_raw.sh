#!/bin/bash
# 在 DAIC 上不方便用 git 时，用 raw.githubusercontent 把 "DreamOn RQ2/ablation +
# Fig4/5 全模型" 这批改动拉到位。
#
# 前提：这些文件必须先在 Mac 上 commit + push 到 GitHub（REF 默认 main），
#       raw 才拉得到。push 后等 ~半分钟（绕过 CDN 缓存本脚本已加 ?t= 时间戳）。
#
# 用法（DAIC，任意目录直接跑）：
#   curl -fsSL https://raw.githubusercontent.com/D4vidHuang/CoRefusion/main/server/jobs/fetch_dreamon_fig45_from_raw.sh | bash
#   # 指定分支/commit：  REF=feat/xxx  curl ... | bash      （或先下脚本再 REF=xxx bash 脚本）
#
# 之后（在仓库根）：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch server/jobs/DreamOn-mask-ablation.slurm --max-samples 20   # 冒烟
#   sbatch server/jobs/DreamOn-RQ2-deobf.slurm    --max-samples 20    # 冒烟
#   sbatch server/jobs/DreamOn-mask-ablation.slurm                    # 全量 -> Fig 5
#   sbatch server/jobs/DreamOn-RQ2-deobf.slurm                        # 全量 -> 表 + Fig 8
set -euo pipefail
PROJ="/tudelft.net/staff-umbrella/CoReFusion/CoRefusion"
REF="${REF:-main}"                                  # 可用 REF=xxx 覆盖分支/commit
RAW="https://raw.githubusercontent.com/D4vidHuang/CoRefusion/$REF"
cd "$PROJ"
mkdir -p experiments experiments/1t5t_exp analysis server/jobs docs logs

# 拉取列表：本次改动的 + 两个作业依赖的未改模块（一并覆盖，保证与 repo 一致）。
FILES=(
  # --- 两个 GPU 作业直接需要 ---
  experiments/experiment_deobfuscation_refineID.py        # RQ2 driver（+DreamOn 引擎）
  experiments/1t5t_exp/part2_dreamon_mask_ablation.py     # DreamOn mask-count ablation（新）
  experiments/benchmark_dreamon.py                        # predict_one（被上面两者复用）
  server/jobs/DreamOn-mask-ablation.slurm                 # 作业（新）
  server/jobs/DreamOn-RQ2-deobf.slurm                     # 作业（新）
  # --- Fig 6 扩散步数扫描（DiffuCoder/DreamCoder）---
  experiments/exp_diffusion_steps_benchmark.py            # steps sweep（registry 已含 DreamCoder）
  server/jobs/DiffStep-sweep.slurm                        # 作业（新，按 MODEL 并行）
  # --- 图/表重生成（可在 DAIC 登录节点或本地跑）---
  analysis/em_by_cardinality.py                           # Fig 4 聚合（新）
  analysis/reproduce_rq2_deobfuscation.py                 # 表 + Fig 8（+DreamOn 自动并入）
  # --- 文档 ---
  docs/dreamon_diffusiongemma_runbook.md
)
TS=$(date +%s)   # cache-buster：绕开 raw 的 ~5 分钟 CDN 缓存
for f in "${FILES[@]}"; do
  if curl -fsSL "$RAW/$f?t=$TS" -o "$f"; then
    echo "[ok]   $f ($(wc -l < "$f") 行)"
  else
    echo "[FAIL] $f" >&2; exit 1
  fi
done
chmod +x server/jobs/DreamOn-mask-ablation.slurm server/jobs/DreamOn-RQ2-deobf.slurm 2>/dev/null || true

# 校验关键改动确实到位（raw 缓存偶尔滞后；命中 WARN 就等几分钟重跑本脚本）。
ok=1
grep -q "dreamon_predict_sites" experiments/experiment_deobfuscation_refineID.py \
  && grep -q '"DreamOn-7B"' experiments/experiment_deobfuscation_refineID.py \
  || { echo "WARN: RQ2 driver 疑似旧版（缺 DreamOn 引擎）" >&2; ok=0; }
grep -q "dreamon_mask_ablation_summary" experiments/1t5t_exp/part2_dreamon_mask_ablation.py \
  || { echo "WARN: mask-ablation 脚本疑似旧版/缺失" >&2; ok=0; }
grep -q "resolve_run" analysis/reproduce_rq2_deobfuscation.py \
  || { echo "WARN: reproduce_rq2 疑似旧版（缺 DreamOn 自动发现）" >&2; ok=0; }
[ "$ok" = 1 ] && echo "OK: DreamOn RQ2/ablation 改动均已就位"

# data/test.csv 体积大、服务器上已有，不走 raw；缺失才提示。
[ -f data/test.csv ] || echo "WARN: 缺 data/test.csv（1000 条 RefineID 基准）" >&2

echo
echo "下一步："
echo "  cd $PROJ && mkdir -p logs"
echo "  sbatch server/jobs/DreamOn-mask-ablation.slurm --max-samples 20   # 冒烟"
echo "  sbatch server/jobs/DreamOn-RQ2-deobf.slurm    --max-samples 20    # 冒烟"
echo "  sbatch server/jobs/DreamOn-mask-ablation.slurm                    # 全量 -> Fig 5"
echo "  sbatch server/jobs/DreamOn-RQ2-deobf.slurm                        # 全量 -> 表 + Fig 8"
