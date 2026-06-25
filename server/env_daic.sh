# CoReFusion — DAIC environment bootstrap (source me, don't execute).
# 用法 / usage（在仓库根）:
#   source server/env_daic.sh
# PORTABLE: PROJECT_DIR 自动 = 本仓库根，所以 clone 到任何路径都能用。
# 缓存/venv/pylibs 默认复用旧 CoReFusion umbrella；要换成别的 umbrella:
#   CF_UMBRELLA=/tudelft.net/staff-umbrella/NA source server/env_daic.sh
# 必须在每个交互式会话和每个 SLURM job 里 source 一次。
#
# WHY: $HOME quota 只有 ~5 MB（DAIC 官方确认），任何写到 $HOME 的 cache 都会爆。
# 这里把 uv / pip / huggingface / torch / matplotlib / tmp 等 cache 全部重定向到
# umbrella 共享盘，并激活 umbrella 上由 uv 建好的项目 venv（torch 也由 uv 装，
# 自带 CUDA 库，所以不再 load DAIC 的 py-torch module）。

# ---- locations (PORTABLE) --------------------------------------------------
# PROJECT_DIR 自动从本脚本位置推断（repo 根 = 本文件的上一级），所以仓库 clone 到
# 任何路径都能用（如 /tudelft.net/staff-umbrella/NA/CoReFusion）。
# 重型依赖（HF 模型缓存 ~50GB / uv venv / pylibs_dgemma）默认复用旧 CoReFusion
# umbrella，避免重下；要全新自包含就 export CF_UMBRELLA=<新 umbrella> 再 source。
_ENV_SELF="${BASH_SOURCE[0]:-$0}"
export PROJECT_DIR="$(cd "$(dirname "$_ENV_SELF")/.." && pwd)"
export UMBRELLA="${CF_UMBRELLA:-/tudelft.net/staff-umbrella/CoReFusion}"
export VENV_DIR="$UMBRELLA/venvs/corefusion"     # legacy（uv 之前的 venv，已弃用）

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

# ---- uv: 二进制 + cache + 解释器 + 工具 + 项目 venv 全部放 umbrella -----------
export UV_INSTALL_DIR="$UMBRELLA/uv/bin"             # uv / uvx 可执行文件
export UV_CACHE_DIR="$UMBRELLA/uv/cache"            # 下载 + wheel 构建缓存（可能很大）
export UV_PYTHON_INSTALL_DIR="$UMBRELLA/uv/python"  # uv 自己装的 python 解释器
export UV_TOOL_DIR="$UMBRELLA/uv/tools"             # `uv tool install` 的隔离环境
export UV_TOOL_BIN_DIR="$UMBRELLA/uv/bin"           # `uv tool` 暴露的命令入口
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv"  # uv 在仓库根建/用这个 venv

mkdir -p "$HF_HUB_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" \
         "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" \
         "$MPLCONFIGDIR" "$TMPDIR" "$UV_INSTALL_DIR" "$UV_CACHE_DIR" \
         "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" 2>/dev/null || true

case ":$PATH:" in *":$UMBRELLA/uv/bin:"*) ;; *) export PATH="$UMBRELLA/uv/bin:$PATH" ;; esac

# 私有/受限模型才需要 token；AISE DreamOn 等 public 模型可不设。
# export HF_TOKEN="hf_xxx"

# ---- python env ------------------------------------------------------------
# 首选本仓库的 .venv；没有就复用旧 CoReFusion umbrella 上现成的 uv venv（这样新
# clone 不必重建环境）。都没有再退回 legacy module-torch + venv/pylibs。
_VENV_ACTIVATED=
for _cand in "$PROJECT_DIR/.venv" "$UMBRELLA/CoRefusion/.venv"; do
    if [ -f "$_cand/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$_cand/bin/activate" || true
        echo "[env_daic] python env: uv venv ($_cand)"
        _VENV_ACTIVATED=1
        break
    fi
done
if [ -z "$_VENV_ACTIVATED" ]; then
    echo "[env_daic] uv .venv 不存在，退回 legacy module-torch 路径。" >&2
    echo "[env_daic] 建议在 login 节点跑一次: bash server/setup_uv_daic.sh" >&2
    if command -v module >/dev/null 2>&1; then
        module purge               2>/dev/null || true
        module load 2025/gpu       2>/dev/null || true
        module load py-torch/2.5.1 2>/dev/null || true
    fi
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate" || true
        echo "[env_daic] python env: legacy venv ($VENV_DIR)"
    elif [ -d "$UMBRELLA/pylibs" ]; then
        export PYTHONPATH="$UMBRELLA/pylibs:${PYTHONPATH:-}"
        echo "[env_daic] python env: legacy pylibs ($UMBRELLA/pylibs)"
    fi
fi

# 进入项目目录（job 里相对路径 data/test.csv 才能找到）。
cd "$PROJECT_DIR" 2>/dev/null || true
