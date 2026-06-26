#!/bin/bash
# [在 Mac 上跑] 把 DAIC 上 DreamOn 的两批结果 rsync 回本地仓库：
#   (1) RQ2 去混淆 (deobfuscation_refineID)  -> 进 Table 3 / Fig 7
#   (2) 1t5t mask 消融 (k=1..5)             -> 进 Fig 5 (fig06)
# 需要：TU Delft VPN 连着 + ~/.ssh/config 里的 `daic` 别名。
# 用法：
#   bash pull_dreamon_results.sh
set -euo pipefail

REMOTE_BASE="daic:/tudelft.net/staff-umbrella/CoReFusion/CoRefusion/results"
LOCAL_BASE="$(cd "$(dirname "$0")" && pwd)/results"

echo "==> (1) DreamOn RQ2 去混淆结果 (deobfuscation_refineID)"
mkdir -p "$LOCAL_BASE/deobfuscation_refineID"
# 只拉 DreamOn 的 per-sample CSV + summary，不动已 pin 的 6/12 DiffuCoder/DreamCoder 文件
rsync -avz --progress \
  --include='DreamOn-7B_all-masked_*.csv' \
  --include='DreamOn-7B_target-only_*.csv' \
  --include='summary_*.csv' \
  --exclude='*' \
  "$REMOTE_BASE/deobfuscation_refineID/" "$LOCAL_BASE/deobfuscation_refineID/"

echo
echo "==> (2) DreamOn 1t5t mask 消融结果 (k=1..5)"
mkdir -p "$LOCAL_BASE/1t5t_exp"
rsync -avz --progress \
  --include='dreamon_mask_ablation_*.csv' \
  --exclude='*' \
  "$REMOTE_BASE/1t5t_exp/" "$LOCAL_BASE/1t5t_exp/"

echo
echo "==> 拉到的 DreamOn 文件："
ls -lt "$LOCAL_BASE/deobfuscation_refineID/"DreamOn-7B_*.csv 2>/dev/null | head
ls -lt "$LOCAL_BASE/1t5t_exp/"dreamon_mask_ablation_*.csv 2>/dev/null | head

echo
echo "==> 下一步：本地重算并重画（用仓库里的 figure venv）"
echo "    # Table 3 + Fig 7（DreamOn 一旦有 CSV 会自动替换掉占位的 partial 数）"
echo "    ./.venv-fig/bin/python analysis/reproduce_rq2_deobfuscation.py"
echo "    # Fig 5（fig06）会自动带上 DreamOn 的 k=1/k=5"
echo "    ./.venv-fig/bin/python figures/rebuilt/fig06_mask_count_ablation.py"
echo "    # 然后把生成的 pdf 复制进论文仓库 figures/（见 README / 上一步输出路径）"
