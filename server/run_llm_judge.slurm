#!/bin/bash
#SBATCH --job-name=rid_judge
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge-%j.out
#SBATCH --error=logs/judge-%j.err
# Consistency-gated LLM-as-Judge，跑 results/unified_refineID/predictions/ 下的
# AR / Seq2Seq（以及其它）模型结果。全 site 一致性门控：只有每个 [MASK] 都吐出
# 同一个非空标识符的样本才送进 judge（拿那个一致名打分），不一致的直接判 0。
# judge 默认 Qwen2.5-7B-Instruct（public，无需 HF_TOKEN）。
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion && mkdir -p logs
#   sbatch server/run_llm_judge.slurm                            # 全部模型
#   sbatch server/run_llm_judge.slurm --only Qwen2.5-Coder-7B    # 单个（可重复）
#   sbatch server/run_llm_judge.slurm --max-samples 50           # 冒烟
#   sbatch server/run_llm_judge.slurm --judge-model Qwen2.5-14B-Instruct
#   sbatch server/run_llm_judge.slurm --resume                   # 断点续判
# 脚本名后的参数原样透传给 run_llm_judge_unified.py。
set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh
mkdir -p logs

echo "=== node:$(hostname) job:${SLURM_JOB_ID:-NA} LLM-as-Judge (all-site consistency gated) ==="
nvidia-smi || true
python -c "import torch,transformers; print('torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())"

srun python experiments/run_llm_judge_unified.py "$@"

echo ""
echo "=== results/unified_refineID/llm_judge/ ==="
ls -lt results/unified_refineID/llm_judge/ 2>/dev/null | head -20
echo "--- latest judge leaderboard ---"
ls -t results/unified_refineID/llm_judge/judge_leaderboard_*.csv 2>/dev/null | head -1 | xargs -r cat
