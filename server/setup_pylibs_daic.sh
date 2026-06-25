#!/bin/bash
# 一次性：把 python 依赖用 pip --target 装到 umbrella 的 pylibs/。
# DAIC 的 module python (spack 'python-venv') 自带的是 *没有 pip* 的精简版，
# 且 ensurepip 坏掉，所以先用 get-pip.py 把 pip bootstrap 到 $UMBRELLA/pytools，
# 再用这个 pip 做 `pip install --target pylibs`。torch 用 module 自带的，不装。
# 必须在 LOGIN 节点跑（要联网），不要在计算节点。
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/setup_pylibs_daic.sh
#
# 之后 env_daic.sh 会自动把 pylibs 加到 PYTHONPATH。
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"

PYTOOLS="$UMBRELLA/pytools"          # pip/setuptools/wheel 装这里（只在安装时用）
GETPIP="$TMPDIR/get-pip.py"
mkdir -p "$PYTOOLS" "$UMBRELLA/pylibs"

# ---- bootstrap pip（module python 没 pip）---------------------------------
if ! PYTHONPATH="$PYTOOLS:${PYTHONPATH:-}" python -m pip --version >/dev/null 2>&1; then
    echo ">> module python 无 pip，用 get-pip.py bootstrap 到 $PYTOOLS"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GETPIP"
    python "$GETPIP" --target "$PYTOOLS"
fi
export PYTHONPATH="$PYTOOLS:${PYTHONPATH:-}"
echo ">> pip: $(python -m pip --version)"

# ---- 依赖装到 pylibs ------------------------------------------------------
echo ">> installing into $UMBRELLA/pylibs ..."
python -m pip install --no-cache-dir --target "$UMBRELLA/pylibs" \
    "transformers==4.57.1" accelerate huggingface_hub tqdm einops \
    sentencepiece protobuf omegaconf

# 用 module 的 numpy（避免 pip 版本与 module torch 的 ABI 冲突）。
rm -rf "$UMBRELLA"/pylibs/numpy* 2>/dev/null || true

# ---- verify（运行时只需 pylibs，不需要 pytools）---------------------------
unset PYTHONPATH
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"
echo ""
echo "=== verify ==="
python - <<'PY'
import torch, transformers, accelerate, huggingface_hub
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("accelerate  ", accelerate.__version__)
print("OK")
PY
