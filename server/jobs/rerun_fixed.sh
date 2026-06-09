#!/bin/bash
# 重新提交三个修复后的失败模型（CodeT5p-2B / CodeT5p-16B / DreamOn-7B）。
#
# 失败原因（logs/ 里 269103/269114/269115 号 job）：
#   CodeT5p-*   transformers 4.57 的 generate() 会无参重建 CodeT5pConfig，
#               触发其断言 "Config has to be initialized with encoder and
#               decoder config"，每个样本都报错。修复 = 加载后调用
#               benchmark_codet5p_16b.ensure_generate_compatible()，
#               并顺手修了 decoder_start_token 前置导致的预测提取偏移。
#   DreamOn-7B  旧版 benchmark_dreamon.py 第 330 行 f-string 表达式里有
#               反斜杠，DAIC 的 Python 3.11 直接 SyntaxError（3.12 才允许），
#               每行 import 失败。修复已提交。
#   另外 CodeT5p-16B walltime 12h -> 30h（无 KV cache 解码较慢）。
#
# 用法（DAIC login 节点）：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   git pull
#   bash server/jobs/rerun_fixed.sh --max-samples 5   # 先冒烟，看 logs/*.out
#   bash server/jobs/rerun_fixed.sh                   # 全量重跑（覆盖坏 CSV）
#   bash server/jobs/aggregate.sh                     # 全部跑完后合并 leaderboard
set -euo pipefail
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
mkdir -p logs
for m in CodeT5p-2B CodeT5p-16B DreamOn-7B; do
  sbatch "server/jobs/${m}.slurm" "$@"
done
squeue -u "$USER"
