#!/bin/bash
# 所有单模型 job 跑完后，重新评分 predictions/ 下所有已存在的模型，
# 生成合并后的 results/unified_refineID/leaderboard.csv。纯 CPU。
set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh
DICT_ARG=""
for d in /usr/share/dict/words "$UMBRELLA/words.txt"; do
  [ -f "$d" ] && DICT_ARG="--dict $d" && break
done
python experiments/run_all_refineID_unified.py --skip-inference $DICT_ARG
echo "---- leaderboard (results/unified_refineID/leaderboard.csv) ----"
cat results/unified_refineID/leaderboard.csv
