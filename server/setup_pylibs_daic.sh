#!/bin/bash
# 一次性:把 python 依赖用 pip --target 装到 umbrella 的 pylibs/ 目录。
# DAIC 的 staff-umbrella(NFS+SELinux)建不了 venv,所以用 --target + PYTHONPATH。
# 必须在 LOGIN 节点跑(要联网),不要在计算节点。
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/setup_pylibs_daic.sh
#
# 之后 env_daic.sh 会自动把 pylibs 加到 PYTHONPATH。torch 用 module 自带的,不装。
set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh

mkdir -p "$UMBRELLA/pylibs"
echo "Installing into $UMBRELLA/pylibs ..."
python -m pip install --no-cache-dir --target "$UMBRELLA/pylibs" \
    "transformers==4.57.1" accelerate huggingface_hub tqdm einops \
    sentencepiece protobuf omegaconf

# 用 module 的 numpy(避免 pip 版本与 module torch 的 ABI 冲突)。
rm -rf "$UMBRELLA"/pylibs/numpy* 2>/dev/null || true

# 本 shell 里 env_daic.sh 是在 pylibs 还不存在时 source 的,这里手动补上再自检。
export PYTHONPATH="$UMBRELLA/pylibs:${PYTHONPATH:-}"
echo ""
echo "=== verify ==="
python - <<'PY'
import torch, transformers, accelerate, huggingface_hub
print("torch       ", torch.__version__, "| cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate  ", accelerate.__version__)
print("OK")
PY
