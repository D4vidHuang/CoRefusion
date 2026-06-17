#!/bin/bash
# 在 DAIC 上不方便用 git 时，用 raw.githubusercontent 把 DiffusionGemma RQ1 所需文件拉到位。
#
# 用法（DAIC，任意目录直接跑）：
#   curl -fsSL https://raw.githubusercontent.com/D4vidHuang/CoRefusion/main/server/jobs/fetch_dgemma_from_raw.sh | bash
#
# 之后：
#   PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh          # 一次性装库+拉权重
#   HF_TOKEN=hf_xxx bash server/jobs/dgemma_rq1_all.sh        # 一键全流程
set -euo pipefail
PROJ="/tudelft.net/staff-umbrella/CoReFusion/CoRefusion"
RAW="https://raw.githubusercontent.com/D4vidHuang/CoRefusion/main"
cd "$PROJ"
mkdir -p experiments server/jobs docs logs

# 拉取列表：改过的 + 本任务必需的（未改的 slurm/setup 也覆盖一遍，保证与 repo 一致）。
FILES=(
  experiments/benchmark_diffusiongemma.py
  server/jobs/DiffusionGemma-smoke.slurm
  server/jobs/dgemma_rq1_all.sh
  server/jobs/DiffusionGemma-26B-A4B.slurm
  server/setup_dgemma_pylibs.sh
  docs/diffusiongemma_runbook.md
  docs/diffusiongemma_rq_scope.md
)
TS=$(date +%s)   # cache-buster：绕开 raw 的 ~5 分钟 CDN 缓存
for f in "${FILES[@]}"; do
  if curl -fsSL "$RAW/$f?t=$TS" -o "$f"; then
    echo "[ok]   $f ($(wc -l < "$f") 行)"
  else
    echo "[FAIL] $f" >&2; exit 1
  fi
done
chmod +x server/jobs/dgemma_rq1_all.sh server/setup_dgemma_pylibs.sh 2>/dev/null || true

# 校验关键改动确实到位（raw 缓存偶尔滞后）。
if grep -q "GATE: in the SLURM chain" experiments/benchmark_diffusiongemma.py; then
  echo "OK: 适配器修复已就位"
else
  echo "WARN: 适配器疑似旧版（raw CDN 缓存？等几分钟重跑本脚本）" >&2
fi
echo
echo "下一步："
echo "  PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh"
echo "  HF_TOKEN=hf_xxx bash server/jobs/dgemma_rq1_all.sh"
