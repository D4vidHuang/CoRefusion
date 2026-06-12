#!/bin/bash
# 一次性安装 DiffusionGemma 专用 python 库（DAIC login 节点执行）。
#
# 为什么单独一棵树：diffusion_gemma 架构 transformers>=5.11.0 才有（5.10.x 没有），
# 而主环境钉死 4.57.1 给 Dream/DreamOn 的自定义代码用，不能升级。所以装到
# $UMBRELLA/pylibs_dgemma，只在 DiffusionGemma 的 job 里把它 PREPEND 到
# PYTHONPATH（PYTHONPATH 先于 venv site-packages，新 transformers 生效）。
#
# 坑（269763 踩过）：env_daic.sh 现在激活 uv venv，而 `uv venv` 默认不带 pip，
# 裸 `pip` 命令会直接挂 → 这里优先用 `uv pip`，回退 `python -m pip`。
#
# 用法：
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/setup_dgemma_pylibs.sh
#   PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh   # 顺便把 ~52GB 权重拉到 hf_cache
set -euo pipefail
source "$(dirname "$0")/env_daic.sh"

DEST="$UMBRELLA/pylibs_dgemma"
TFM_VER="5.11.0"   # 第一个含 diffusion_gemma 的 release（2026-06-10）

# 清掉上次失败留下的半成品/空目录，从干净状态装。
rm -rf "$DEST"
mkdir -p "$DEST"

# torch 不装（用 venv 里已有的 2.12+cu130）；accelerate 给 device_map="auto"；
# pillow 是 AutoProcessor（多模态）import 需要；其余依赖让 transformers 自己拉。
PKGS=("transformers==${TFM_VER}" accelerate pillow tqdm sentencepiece protobuf)
if command -v uv >/dev/null 2>&1; then
  uv pip install --no-cache --target "$DEST" "${PKGS[@]}"
elif python -m pip --version >/dev/null 2>&1; then
  python -m pip install --no-cache-dir --target "$DEST" "${PKGS[@]}"
else
  echo "FATAL: 既没有 uv 也没有 pip（uv venv 默认无 pip）。" >&2
  echo "  装一个：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# 与主 pylibs 同样的约定：删掉自带 numpy，用 venv 的 numpy（torch 按它编译）。
rm -rf "$DEST"/numpy "$DEST"/numpy-* "$DEST"/numpy.libs 2>/dev/null || true

# torchvision（Gemma4 图像处理器要它；纯文本不装也行，代码会回退 AutoTokenizer）。
# 版本要和 venv 的 torch 配对（0.X = torch minor + 15），且从同一 cu 索引装，
# --no-deps 防止拖进第二份 torch 经 PYTHONPATH 盖住 venv 的 cu130 版。
read -r TV_VER TORCH_CU <<<"$(python -c "import torch; v=torch.__version__; print('0.%d' % (int(v.split('.')[1])+15), (v.split('+')[1] if '+' in v else 'none'))")"
TV_IDX=()
case "$TORCH_CU" in cu*) TV_IDX=(--index-url "https://download.pytorch.org/whl/${TORCH_CU}");; esac
if command -v uv >/dev/null 2>&1; then
  uv pip install --no-cache --target "$DEST" --no-deps "${TV_IDX[@]}" "torchvision==${TV_VER}.*" \
    || echo "WARN: torchvision==${TV_VER}.* 装不上，跳过（纯文本 fallback 不受影响）"
else
  python -m pip install --no-cache-dir --target "$DEST" --no-deps "${TV_IDX[@]}" "torchvision==${TV_VER}.*" \
    || echo "WARN: torchvision==${TV_VER}.* 装不上，跳过（纯文本 fallback 不受影响）"
fi

PYTHONPATH="$DEST" python - <<'EOF'
import transformers, importlib
print("transformers", transformers.__version__, "@", transformers.__file__)
importlib.import_module("transformers.models.diffusion_gemma")
print("OK: diffusion_gemma architecture available")
try:
    import torchvision
    print("torchvision", torchvision.__version__, "OK (AutoProcessor usable)")
except Exception as ex:
    print("torchvision unavailable (%s) -> AutoTokenizer text-only fallback" % str(ex)[:80])
EOF

# 模型公开（Apache 2.0，不 gated），无需 HF_TOKEN。
if [ "${PREDOWNLOAD:-0}" = "1" ]; then
  PYTHONPATH="$DEST" python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/diffusiongemma-26B-A4B-it')
print('predownload done')"
fi

echo "done -> $DEST"
