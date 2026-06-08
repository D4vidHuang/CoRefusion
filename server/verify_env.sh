#!/bin/bash
# 在 DAIC 的 LOGIN node 上运行，确认环境。把输出贴回来即可定稿。
#   bash /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/verify_env.sh
UMBRELLA="/tudelft.net/staff-umbrella/CoReFusion"

echo "### whoami / host";            whoami; hostname
echo; echo "### home quota";         quota -s 2>/dev/null || quota 2>/dev/null || echo "(quota n/a)"
echo; echo "### umbrella df";        df -h "$UMBRELLA" 2>&1
echo; echo "### repo + data present?"
ls -la "$UMBRELLA/CoRefusion/data/test.csv" 2>&1
ls -la "$UMBRELLA/CoRefusion/experiments/benchmark_dreamon_java_identifiers.py" 2>&1
echo; echo "### my SLURM account(s) (REQUIRED for sbatch)"
sacctmgr -nP show assoc user="$USER" format=account 2>&1 | head
echo; echo "### SLURM partitions (expect 'all')"
sinfo -s 2>&1 | head -40
echo; echo "### GPU gres per partition"
sinfo -o "%P %.11l %.6D %G" 2>&1 | head -50
echo; echo "### DAIC Lmod: GPU base + torch availability"
( module purge; module load 2025/gpu; module avail py-torch ) 2>&1 | tr '\t' '\n' | grep -iE "torch|2025|cuda" | head -30 \
  || echo "(module n/a)"
echo; echo "### done — paste everything above back"
