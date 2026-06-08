#!/bin/bash
# One-time environment setup on DAIC. Builds a venv ON THE UMBRELLA (never in
# $HOME) on top of the cluster's py-torch/2.5.1 module, then pip-installs the
# small extras (transformers 4.57.1, etc.).
#
#   bash /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/setup_daic.sh
#
# 建议在交互节点上跑（venv 创建/小包安装很快，但守规矩）：
#   sinteractive --cpus-per-task=4 --mem=16G --time=01:00:00   # 或:
#   srun --partition=all --account=<acct> --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env_daic.sh"     # loads modules; venv may not exist yet

echo ">> module python: $(command -v python)"
python -c "import torch; print('module torch:', torch.__version__, '| cuda build:', torch.version.cuda)" \
    || { echo "ERROR: py-torch module not loaded. Run \`module avail py-torch\` and fix env_daic.sh"; exit 1; }

# 1) venv on the umbrella, reusing module torch via --system-site-packages ----
# DAIC 的 module python (spack 'python-venv') 的 ensurepip 是坏的,所以用
# --without-pip 建 venv,再手动 bootstrap pip,绕开 ensurepip。
if [ -d "$VENV_DIR" ] && [ ! -x "$VENV_DIR/bin/python" ]; then
    echo ">> nuking half-built venv from a previous run (NFS leftovers)"
    rm -rf "$VENV_DIR" "$(dirname "$VENV_DIR")"/.trash_* 2>/dev/null || true
    # if a file refuses to delete (NFS), move the dir aside so the path is clean
    [ -e "$VENV_DIR" ] && mv "$VENV_DIR" "$(dirname "$VENV_DIR")/.trash_$$" 2>/dev/null || true
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo ">> creating venv at $VENV_DIR (--without-pip --copies, NFS-safe)"
    mkdir -p "$(dirname "$VENV_DIR")"
    python -m venv "$VENV_DIR" --system-site-packages --without-pip --copies
fi
source "$VENV_DIR/bin/activate"
echo ">> venv python: $(command -v python)  ($(python --version))"

# bootstrap pip into the venv if it isn't already importable
if ! python -m pip --version >/dev/null 2>&1; then
    echo ">> bootstrapping pip via get-pip.py"
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"
    python "$TMPDIR/get-pip.py"
fi
python -m pip --version

# 2) deps (transformers pinned; torch stays the module one) ------------------
python -m pip install --upgrade pip
python -m pip install -r "$HERE/requirements.txt"

# 3) run dirs (in the repo, i.e. on the umbrella) ---------------------------
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results/dreamon_java" "$PROJECT_DIR/hf_models"

# 4) sanity -----------------------------------------------------------------
python - <<'PY'
import torch, transformers
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda build  :", torch.version.cuda)
print("cuda avail  :", torch.cuda.is_available(), "(False on login node is normal)")
PY

echo ""
echo ">> setup done. venv = $VENV_DIR"
echo ">> next: cd $PROJECT_DIR && sbatch --account=<acct> server/smoke_test.slurm"
