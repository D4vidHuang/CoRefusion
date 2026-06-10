#!/bin/bash
# DreamOn Java-Identifier 微调模型 benchmark —— 两种 __MASKED_VAR__ mask-style 对比。
#
# 背景：AISE-TUDelft/dreamon-{0.5b,7b}-Java-Identifiers 微调时用 __MASKED_VAR__
# 标记变量站点（字面文本，不是 tokenizer token；diffusion 仍只在 <|mask|> 位置
# 生成）。训练 prompt 的确切排布没有记录，所以两种合理排布都实现了：
#   canvas      每个站点放 len(tokenize("__MASKED_VAR__"))=5 个 <|mask|>（原位生成）
#   placeholder 每窗口第一个站点放 canvas，其余站点保留字面 __MASKED_VAR__ 作上下文
# 先冒烟对比两种 style 的输出质量，再用胜出的 style 跑全量。
#
# 用法（DAIC login 节点）：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/jobs/dreamon_java_styles.sh            # 冒烟：2 模型 x 2 style x 20 样本（4 个 job）
#   # 看 logs/bench-*.out 里 [debug] 的 preds 是否为合理标识符，选定 style 后全量：
#   FULL=1 STYLE=canvas bash server/jobs/dreamon_java_styles.sh
#   FULL=1 STYLE=placeholder MODELS="dreamon-7b-Java" bash server/jobs/dreamon_java_styles.sh
#   # 其余参数透传 benchmark 脚本（如 --hf-repo ...）
#
# 结果：results/dreamon_java/<Label>_style-<style>_per_{site,sample}_<ts>.csv
set -euo pipefail
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
mkdir -p logs

MODELS="${MODELS:-dreamon-0.5b-Java dreamon-7b-Java}"

if [ "${FULL:-0}" = "1" ]; then
  STYLE="${STYLE:-canvas}"
  for m in $MODELS; do
    sbatch server/run_benchmark.slurm --model "$m" --mask-style "$STYLE" "$@"
  done
else
  for m in $MODELS; do
    for s in canvas placeholder; do
      sbatch server/run_benchmark.slurm --model "$m" --mask-style "$s" --max-samples 20 --debug "$@"
    done
  done
fi
squeue -u "$USER"
