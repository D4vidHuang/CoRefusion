#!/bin/bash
#SBATCH --job-name=rid_gt_judge
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gt_judge-%j.out
#SBATCH --error=logs/gt_judge-%j.err
# LLM-as-Judge 跑 RefineID 数据集“自己的 ground truth”，量化 judge 对金标准
# 标识符的认可率（judge ceiling / 验证 exact-match 自动判 1 的假设）。
# 用和正式 benchmark 完全相同的 judge：模型、SYSTEM_PROMPT、user prompt、
# extract_context、judge_one 全部复用 experiments/llm_judge_variable_naming.py。
# 唯一区别：把每条样本的 ground_truth 当作 prediction 真正送进 LLM（不再因为
# exact-match 就跳过 LLM 直接判 1）。judge 默认 Qwen2.5-7B-Instruct（public）。
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch server/run_llm_judge_groundtruth.slurm                       # 默认 judge
#   sbatch server/run_llm_judge_groundtruth.slurm --max-samples 50      # 冒烟
#   sbatch server/run_llm_judge_groundtruth.slurm --judge-model Qwen2.5-14B-Instruct
#   sbatch server/run_llm_judge_groundtruth.slurm --resume              # 断点续判
# 脚本名后的参数原样透传给 judge_refineid_ground_truth.py。
#
# clean-context 上界（把所有 [MASK] 都填成 gt，去掉残留 [MASK] 噪声）：
#   sbatch server/run_llm_judge_groundtruth.slurm --mask-fill all
#
# 多尺寸 judge（--judge-model 可重复，单 job 内顺序加载/卸载）：
#   sbatch server/run_llm_judge_groundtruth.slurm \
#          --judge-model Qwen2.5-7B-Instruct --judge-model Qwen2.5-14B-Instruct
#   注意：默认 --time=08:00:00 对单个 7B judge（1000 条、max_new_tokens=32）绰绰有余；
#   一个 job 里串多个大 judge 时用 sbatch --time=24:00:00 覆盖，或超时后 --resume 续跑。
# 大 judge（24B/27B/32B，bf16 47-65GB）放不进 48GB A40，要指定 96GB 卡：
#   sbatch --gres=gpu:<GRES型名>:1 server/run_llm_judge_groundtruth.slurm \
#          --judge-model Qwen2.5-32B-Instruct
#   （GRES 型名用  sinfo -o "%N %G" | grep -i 6000  查询）
set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh
mkdir -p logs

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} LLM-as-Judge on RefineID ground truth ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

srun python experiments/judge_refineid_ground_truth.py "$@"

echo ""
echo "=== results/refineid_groundtruth_judge/ ==="
ls -lt results/refineid_groundtruth_judge/ 2>/dev/null | head -20
echo "--- latest summary ---"
ls -t results/refineid_groundtruth_judge/groundtruth_summary_*.csv 2>/dev/null | head -1 | xargs -r cat
