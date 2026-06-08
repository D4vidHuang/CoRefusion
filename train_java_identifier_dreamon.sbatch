#!/usr/bin/env bash
#SBATCH --job-name=dreamon-var-rename-7b
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

RUN_SCRIPT="scripts/run_java_identifier_dreamon_a100.sh"

if [[ -n "${REPO_ROOT:-}" ]]; then
  cd "${REPO_ROOT}"
elif [[ -f "${RUN_SCRIPT}" && -d "slurm" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/${RUN_SCRIPT}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/../${RUN_SCRIPT}" ]]; then
    cd "${SCRIPT_DIR}/.."
  else
    echo "Could not locate DreamOn repo root." >&2
    echo "Submit from the repo root or set REPO_ROOT to the checkout path." >&2
    exit 1
  fi
fi
REPO_ROOT="."
DEFAULT_SIF="artifacts/dreamon-java-a100_20260519.sif"
if [[ ! -f "${DEFAULT_SIF}" && -f "apptainer/dreamon-java-a100.sif" ]]; then
  DEFAULT_SIF="apptainer/dreamon-java-a100.sif"
fi
SIF="${SIF:-${DEFAULT_SIF}}"

if [[ ! -f "${SIF}" ]]; then
  echo "Missing Apptainer image: ${SIF}" >&2
  echo "Checked from repo directory: $(pwd)" >&2
  echo "Set SIF to an image path or build apptainer/dreamon-java-a100.sif" >&2
  exit 1
fi

gpu_count_from_csv() {
  local csv="$1"
  [[ -n "${csv}" ]] || return 1
  local old_ifs="${IFS}"
  IFS=','
  read -r -a values <<<"${csv}"
  IFS="${old_ifs}"
  echo "${#values[@]}"
}

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    NPROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE="$(gpu_count_from_csv "${CUDA_VISIBLE_DEVICES}")"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    NPROC_PER_NODE="$(gpu_count_from_csv "${SLURM_JOB_GPUS}")"
  else
    NPROC_PER_NODE=3
  fi
fi

if [[ -z "${GPU_IDS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_IDS="${CUDA_VISIBLE_DEVICES}"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    GPU_IDS="${SLURM_JOB_GPUS}"
  else
    GPU_IDS="$(seq -s, 0 "$((NPROC_PER_NODE - 1))")"
  fi
fi

HF_HOME="${HF_HOME:-.cache/huggingface}"
mkdir -p "${HF_HOME}"
export HF_HOME
export APPTAINERENV_HF_HOME="${APPTAINERENV_HF_HOME:-${HF_HOME}}"

HF_AUTH_SOURCE=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_AUTH_SOURCE="HF_TOKEN"
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  HF_AUTH_SOURCE="HUGGING_FACE_HUB_TOKEN"
elif [[ -n "${HF_TOKEN_FILE:-}" ]]; then
  if [[ ! -f "${HF_TOKEN_FILE}" ]]; then
    echo "HF_TOKEN_FILE is set but does not exist: ${HF_TOKEN_FILE}" >&2
    exit 1
  fi
  HF_TOKEN="$(< "${HF_TOKEN_FILE}")"
  HF_AUTH_SOURCE="HF_TOKEN_FILE"
elif [[ -n "${HF_TOKEN_PATH:-}" ]]; then
  if [[ ! -f "${HF_TOKEN_PATH}" ]]; then
    echo "HF_TOKEN_PATH is set but does not exist: ${HF_TOKEN_PATH}" >&2
    exit 1
  fi
  HF_TOKEN="$(< "${HF_TOKEN_PATH}")"
  HF_AUTH_SOURCE="HF_TOKEN_PATH"
elif [[ -f "${HF_HOME}/token" ]]; then
  HF_TOKEN="$(< "${HF_HOME}/token")"
  HF_AUTH_SOURCE="HF_HOME/token"
elif [[ -n "${HOME:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
  HF_AUTH_SOURCE="~/.cache/huggingface/token"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="${HF_TOKEN//$'\n'/}"
  HF_TOKEN="${HF_TOKEN//$'\r'/}"
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "Hugging Face token from ${HF_AUTH_SOURCE} is empty after trimming whitespace." >&2
    exit 1
  fi
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
  export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  echo "HF auth: token provided via ${HF_AUTH_SOURCE}"
else
  echo "HF auth: no token provided; set HF_TOKEN or HF_TOKEN_FILE for gated/private Hub resources."
fi

if [[ -n "${EXTRA_BIND_PATHS:-}" ]]; then
  APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}${EXTRA_BIND_PATHS}"
  export APPTAINER_BINDPATH
fi

export GPU_IDS
export NPROC_PER_NODE
export MASTER_PORT="${MASTER_PORT:-$((12000 + ${SLURM_JOB_ID:-0} % 20000))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export DATA_DIR="${DATA_DIR:-data/stackv2-java-identifier-large-1024}"
export MODEL_PATH="${MODEL_PATH:-checkpoints/DreamOn-v0-7B}"
export CKPT_DIR="${CKPT_DIR:-checkpoints/dreamcoder-7b-variable-renaming}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export MAX_IDENTIFIER_TOKENS="${MAX_IDENTIFIER_TOKENS:-10}"
# Global batch must be divisible by 3 GPUs. Keep DreamOn's micro batch of 8.
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10000}"
export SAVE_CHECKPOINT_STEPS="${SAVE_CHECKPOINT_STEPS:-100}"
export LR="${LR:-1e-5}"
export WARMUP_STEPS_RATIO="${WARMUP_STEPS_RATIO:-0.1}"
export VAL_STEPS="${VAL_STEPS:-10}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SKIP_FINAL_CHECKPOINT="${SKIP_FINAL_CHECKPOINT:-false}"
export PROGRESS_BAR="${PROGRESS_BAR:-false}"
export PROGRESS_LOG_STEPS="${PROGRESS_LOG_STEPS:-10}"
export FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-FULL_SHARD}"
export FSDP_SYNC_MODULE_STATES="${FSDP_SYNC_MODULE_STATES:-true}"
export FSDP_USE_ORIG_PARAMS="${FSDP_USE_ORIG_PARAMS:-true}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-auto}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-true}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
export USE_WANDB="${USE_WANDB:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="${WANDB_PROJECT:-CoReFusion}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-dreamcoder-7b-variable-renaming}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-dreamcoder-7b-variable-renaming-${RUN_STAMP}}"
export WANDB_TAGS="${WANDB_TAGS:-[java,identifier,variable-renaming,dreamon,dreamcoder,7b]}"

if [[ "${USE_WANDB,,}" == "true" && "${WANDB_MODE,,}" == "online" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -n "${WANDB_API_KEY_FILE:-}" ]]; then
    if [[ ! -f "${WANDB_API_KEY_FILE}" ]]; then
      echo "WANDB_API_KEY_FILE is set but does not exist: ${WANDB_API_KEY_FILE}" >&2
      exit 1
    fi
    WANDB_API_KEY="$(< "${WANDB_API_KEY_FILE}")"
    WANDB_API_KEY="${WANDB_API_KEY//$'\n'/}"
    WANDB_API_KEY="${WANDB_API_KEY//$'\r'/}"
    export WANDB_API_KEY
  fi

  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
    export APPTAINERENV_WANDB_MODE="${WANDB_MODE}"
    echo "Logging in to W&B inside ${SIF}"
    apptainer exec --nv "${SIF}" wandb login --relogin "${WANDB_API_KEY}"
  else
    echo "USE_WANDB=true and WANDB_MODE=online, but no W&B API key was provided." >&2
    echo "Set WANDB_API_KEY or WANDB_API_KEY_FILE before sbatch." >&2
    exit 1
  fi
fi

echo "Slurm job ${SLURM_JOB_ID:-local}: node=${SLURMD_NODENAME:-unknown}, GPUs=${GPU_IDS}, ranks=${NPROC_PER_NODE}"
echo "SIF=${SIF}"
echo "Repo=${REPO_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "Model=${MODEL_PATH}"
echo "Data=${DATA_DIR}"
echo "Checkpoints=${CKPT_DIR}"
echo "DreamOn variable-renaming recipe: lr=${LR}, batch=${TRAIN_BATCH_SIZE}, micro=${MICRO_BATCH_SIZE_PER_GPU}, max_length=${MAX_LENGTH}, steps=${TOTAL_TRAINING_STEPS}"
echo "Runtime: backend=${DISTRIBUTED_BACKEND}, sharding=${FSDP_SHARDING_STRATEGY}, attention=${ATTN_IMPLEMENTATION}, grad_ckpt=${ENABLE_GRADIENT_CHECKPOINTING}, remove_padding=${USE_REMOVE_PADDING}"
echo "Progress: bar=${PROGRESS_BAR}, log_steps=${PROGRESS_LOG_STEPS}"
echo "W&B: enabled=${USE_WANDB}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}, run=${WANDB_RUN_NAME}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  apptainer exec --nv "${SIF}" \
  bash "${RUN_SCRIPT}" "$@"
