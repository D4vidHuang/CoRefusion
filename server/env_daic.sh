# CoReFusion — DAIC environment bootstrap (source me, don't execute).
# 用法 / usage:
#   source /tudelft.net/staff-umbrella/CoReFusion/CoRefusion/server/env_daic.sh
# 必须在每个交互式会话和每个 SLURM job 里 source 一次。
#
# WHY: $HOME quota 只有 ~5 MB（DAIC 官方确认），任何写到 $HOME 的 cache 都会爆。
# 这里把 pip / huggingface / torch / matplotlib / tmp 等 cache 全部重定向到
# umbrella 共享盘，并按 DAIC 的 Lmod 层级加载 py-torch，再激活 umbrella 上的 venv。

# ---- locations -------------------------------------------------------------
export UMBRELLA="/tudelft.net/staff-umbrella/CoReFusion"
export PROJECT_DIR="$UMBRELLA/CoRefusion"
export VENV_DIR="$UMBRELLA/venvs/corefusion"

# ---- redirect ALL caches off $HOME ----------------------------------------
export HF_HOME="$UMBRELLA/hf_cache"                 # huggingface_hub / transformers
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"         # legacy alias, harmless
export PIP_CACHE_DIR="$UMBRELLA/pip_cache"
export TORCH_HOME="$UMBRELLA/cache/torch"
export XDG_CACHE_HOME="$UMBRELLA/cache"
export XDG_CONFIG_HOME="$UMBRELLA/config"
export XDG_DATA_HOME="$UMBRELLA/data_home"
export TRITON_CACHE_DIR="$UMBRELLA/cache/triton"
export MPLCONFIGDIR="$UMBRELLA/config/matplotlib"
export TMPDIR="$UMBRELLA/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p "$HF_HUB_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" \
         "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" \
         "$MPLCONFIGDIR" "$TMPDIR" 2>/dev/null || true

# 私有/受限模型才需要 token；AISE DreamOn 模型是 public，可不设。
# export HF_TOKEN="hf_xxx"

# ---- DAIC modules (Lmod hierarchy) ----------------------------------------
# 先加载 GPU base，再加载预编译好的 torch 2.5.1（已含 cuda/12.9）。
# venv 用 --system-site-packages 创建，所以必须在激活 venv 前加载同一套 module。
if command -v module >/dev/null 2>&1; then
    module purge                  2>/dev/null || true
    module load 2025/gpu          2>/dev/null || true
    module load py-torch/2.5.1    2>/dev/null || true
fi

# ---- venv on the umbrella --------------------------------------------------
# venv 里装的是我们自己的 transformers==4.57.1 等，会 shadow 掉 module 自带版本。
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# 进入项目目录（job 里相对路径 data/test.csv 才能找到）。
cd "$PROJECT_DIR" 2>/dev/null || true
