#!/bin/bash
# 一次性：用 uv 在仓库根建 .venv（在 umbrella 上）并装好 AR 推理依赖。
# torch 用 PyPI 的 CUDA wheel（自带 cuda 库，不依赖 DAIC 的 py-torch module）。
# 必须在 LOGIN 节点跑（要联网；torch wheel ~2GB，会落到 UV_CACHE_DIR=umbrella）。
#
#   cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
#   bash server/setup_uv_daic.sh
#
# 之后 env_daic.sh 会自动激活 $PROJECT_DIR/.venv。
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/server/env_daic.sh"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: 找不到 uv。先把 uv 装到 umbrella：" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=\"\$UMBRELLA/uv/bin\" INSTALLER_NO_MODIFY_PATH=1 sh" >&2
    exit 1
}
echo ">> uv: $(uv --version)  ($(command -v uv))"

cd "$PROJECT_DIR"

# 1) venv：uv 自管 python 3.11，解释器与 venv 全在 umbrella 上 -----------------
uv venv --python 3.11 "$PROJECT_DIR/.venv"
# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv/bin/activate"

# 2) 依赖：torch 用 cu128 wheel（含 Blackwell sm_120 内核，同时兼容 A40 sm_86），
#    这样 gpu:1 落到 A40 或 RTX PRO 6000 都能跑。再装其余依赖。 ------------------
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==4.57.1" accelerate huggingface_hub tqdm einops \
               sentencepiece protobuf omegaconf pandas
# pandas：benchmark_diffusion_models.py 顶层 import 它（DiffuCoder/DreamCoder 这条 engine 必需）。

# 3) sanity -----------------------------------------------------------------
python - <<'PY'
import torch, transformers, accelerate
print("torch       ", torch.__version__, "| cuda-build", torch.version.cuda)
print("transformers", transformers.__version__)
print("accelerate  ", accelerate.__version__)
print("cuda avail  ", torch.cuda.is_available(), "(login 节点显示 False 正常，GPU 节点才 True)")
print("OK")
PY

echo ""
echo ">> uv env ready -> $PROJECT_DIR/.venv"
echo ">> next: cd $PROJECT_DIR && mkdir -p logs && sbatch server/run_ar_benchmark.slurm --max-samples 20"
