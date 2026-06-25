#!/bin/bash
# [在 Mac 上跑] 把 DAIC 上的 "LJ 判 RefineID ground truth" 结果全部 rsync 回本地仓库。
# 需要：TU Delft VPN 连着 + ~/.ssh/config 里的 `daic` 别名（已确认存在）。
# 用法：
#   bash pull_gt_judge_results.sh        # 拉 gt-judge 结果
# 第一次会让你输一次密码（ControlMaster 之后复用，不会反复问）。
set -euo pipefail

REMOTE="daic:/tudelft.net/staff-umbrella/CoReFusion/CoRefusion/results/refineid_groundtruth_judge/"
LOCAL="$(cd "$(dirname "$0")" && pwd)/results/refineid_groundtruth_judge/"

mkdir -p "$LOCAL"
echo "==> rsync 从 DAIC 拉结果"
echo "    远端: $REMOTE"
echo "    本地: $LOCAL"
rsync -avz --progress "$REMOTE" "$LOCAL"

echo
echo "==> 拉到的文件："
ls -lt "$LOCAL" | head -40
echo
echo "==> per-sample CSV 数 / summary CSV 数："
ls "$LOCAL"groundtruth__judge_*__*.csv 2>/dev/null | wc -l | xargs echo "  per-sample:"
ls "$LOCAL"groundtruth_summary_*.csv  2>/dev/null | wc -l | xargs echo "  summary:  "
