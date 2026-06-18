#!/bin/bash
# RefineID ground-truth 的 judge ceiling，跑和正式 benchmark 完全相同的 5-judge
# 面板（每个 judge 一个独立 GPU job，并行），全部结束后用 --combine-only 合并成
# 跨 judge 的接受率表。等价于 server/jobs/judge_multi_size.sh，只是把目标脚本换成
# server/run_llm_judge_groundtruth.slurm（判 gt 而不是判模型预测）。
#
# judge 面板（默认 5 个，尺寸阶梯 7B→14B→24B→27B→32B）：
#   Qwen2.5-7B / Qwen2.5-14B            bf16 <30GB，A40 也能跑
#   Mistral-Small-24B / Gemma-2-27B-It / Qwen2.5-32B
#                                        bf16 47-65GB，必须 96GB 卡（RTX PRO 6000）
#
# 用法（DAIC login 节点）：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   # 指定 96GB 卡的 GRES 型名（用 sinfo -o "%N %G" | grep -i 6000 查询）：
#   GPU_TYPE=<GRES型名> HF_TOKEN=hf_xxx bash server/jobs/gt_judge_multi.sh
#   # 不设 GPU_TYPE 则 --gres=gpu:1 任意卡：小 judge 没问题，大 judge 落到 A40 会 OOM
#
#   自定义 judge 集合 / 透传参数（如 --max-samples 冒烟、--mask-fill all 干净上界）：
#   JUDGES="Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct" bash server/jobs/gt_judge_multi.sh --max-samples 50
#   bash server/jobs/gt_judge_multi.sh --resume
#   bash server/jobs/gt_judge_multi.sh --mask-fill all
#
#   Gemma-2-27B-It 是 gated 模型：先 export HF_TOKEN=hf_xxx（无 token 自动跳过）。
#
#   全部 job 结束后（squeue 清空），login 节点直接合并（纯 CPU 不用 sbatch）：
#   source server/env_daic.sh
#   python experiments/judge_refineid_ground_truth.py --combine-only
set -euo pipefail
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
mkdir -p logs

DEFAULT_JUDGES="Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Mistral-Small-24B Gemma-2-27B-It Qwen2.5-32B-Instruct"
JUDGES="${JUDGES:-$DEFAULT_JUDGES}"
GPU_TYPE="${GPU_TYPE:-}"

for j in $JUDGES; do
  if [ "$j" = "Gemma-2-27B-It" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "[skip] $j 需要 HF_TOKEN（gated 模型），先 export HF_TOKEN=hf_xxx 再重跑"
    continue
  fi
  # shellcheck disable=SC2086
  sbatch ${GPU_TYPE:+--gres=gpu:${GPU_TYPE}:1} \
         ${HF_TOKEN:+--export=ALL,HF_TOKEN="$HF_TOKEN"} \
         server/run_llm_judge_groundtruth.slurm --judge-model "$j" "$@"
done
squeue -u "$USER"
echo
echo "全部结束后合并："
echo "  source server/env_daic.sh && python experiments/judge_refineid_ground_truth.py --combine-only"
