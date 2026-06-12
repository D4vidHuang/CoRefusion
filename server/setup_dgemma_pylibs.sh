#!/bin/bash
# 一次性安装 DiffusionGemma 专用 python 库（DAIC login 节点执行）。
#
# 为什么单独一棵树：diffusion_gemma 架构只在最新 transformers 里有，而
# $UMBRELLA/pylibs 钉死 transformers==4.57.1 给 Dream/DreamOn 的自定义代码用，
# 不能升级。所以装到 $UMBRELLA/pylibs_dgemma，只在 DiffusionGemma 的 job 里
# 把它 PREPEND 到 PYTHONPATH（先于旧 pylibs，新 transformers 生效）。
#
# 用法：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/setup_dgemma_pylibs.sh
#   PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh   # 顺便把 ~52GB 权重拉到 hf_cache
set -euo pipefail
source "$(dirname "$0")/env_daic.sh"

DEST="$UMBRELLA/pylibs_dgemma"
mkdir -p "$DEST"

# torch 不装（用运行时已有的 2.12+cu130）；accelerate 给 device_map="auto"；
# pillow 是 AutoProcessor（多模态）import 需要。
pip install --no-cache-dir --target "$DEST" -U \
    transformers accelerate huggingface_hub tokenizers safetensors pillow \
    tqdm sentencepiece protobuf

# 与主 pylibs 同样的约定：删掉自带 numpy，用 module 的 numpy（避免 ABI 冲突）。
rm -rf "$DEST"/numpy "$DEST"/numpy-* "$DEST"/numpy.libs 2>/dev/null || true

PYTHONPATH="$DEST" python - <<'EOF'
import transformers, importlib
print("transformers", transformers.__version__)
try:
    importlib.import_module("transformers.models.diffusion_gemma")
    print("OK: diffusion_gemma architecture available")
except Exception as ex:
    raise SystemExit(
        "FAIL: this transformers has no diffusion_gemma (%s). "
        "Try: pip install --target ... git+https://github.com/huggingface/transformers" % ex)
EOF

# 模型公开（Apache 2.0，不 gated），无需 HF_TOKEN。
if [ "${PREDOWNLOAD:-0}" = "1" ]; then
  PYTHONPATH="$DEST" python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/diffusiongemma-26B-A4B-it')
print('predownload done')"
fi

echo "done -> $DEST"
