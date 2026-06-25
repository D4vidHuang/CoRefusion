#!/bin/bash
# 一键 git 下载/更新整个 CoRefusion 仓库（任意能用 git 的机器：DelftBlue / DAIC / 本地）。
# 仓库是公开的，无需 token。
#
#   bash server/clone_or_update.sh [目标目录]       # 默认 $HOME/CoRefusion
#   DEPTH=1 bash server/clone_or_update.sh /scratch/$USER/CoRefusion   # 浅克隆(只要代码,快)
#   BRANCH=main bash server/clone_or_update.sh ...  # 指定分支
#
# 首次下载也可直接：git clone https://github.com/D4vidHuang/CoRefusion.git
#
# 注意：DelftBlue 的运行环境(module/account/partition/venv)与 DAIC 不同。本脚本只负责
# 把代码弄到位；跑作业前还需 DelftBlue 专属的 env + SBATCH 头(--account/--partition)。
set -euo pipefail
REPO_URL="https://github.com/D4vidHuang/CoRefusion.git"
BRANCH="${BRANCH:-main}"
DEST="${1:-$HOME/CoRefusion}"

if [ -d "$DEST/.git" ]; then
  echo "更新已有仓库: $DEST"
  git -C "$DEST" fetch origin "$BRANCH"
  git -C "$DEST" checkout "$BRANCH"
  git -C "$DEST" pull --ff-only origin "$BRANCH"
else
  echo "clone -> $DEST"
  if [ "${DEPTH:-0}" = "1" ]; then
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$DEST"
  else
    git clone --branch "$BRANCH" "$REPO_URL" "$DEST"
  fi
fi
echo "done -> $DEST  @ $(git -C "$DEST" rev-parse --short HEAD)"
