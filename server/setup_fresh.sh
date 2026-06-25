#!/bin/bash
# FRESH, self-contained env for a machine/user WITHOUT access to the original
# CoReFusion umbrella (so the reused venv + 50GB model cache are unreachable).
# Run ONCE on a LOGIN node (needs internet). Puts everything under CF_UMBRELLA.
#
#   CF_UMBRELLA=<a dir YOU can write to>  bash server/setup_fresh.sh
#   e.g.  CF_UMBRELLA=/tudelft.net/staff-umbrella/NA  bash server/setup_fresh.sh
#
# It: installs uv under $CF_UMBRELLA, builds <repo>/.venv with torch +
# transformers 4.57.1 + deps, redirects all caches under $CF_UMBRELLA, and
# pre-downloads DreamCoder-7B so the (offline) GPU node can load it. That's
# enough to run the DreamCoder diffusion-step sweep.
#
# DiffusionGemma needs the heavier dgemma tree (transformers 5.11 + ~52GB) and a
# 96GB GPU -- see the note printed at the end; better run on the main account.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

: "${CF_UMBRELLA:?ERROR: set CF_UMBRELLA to a writable dir, e.g. CF_UMBRELLA=/tudelft.net/staff-umbrella/NA}"
export CF_UMBRELLA
mkdir -p "$CF_UMBRELLA" 2>/dev/null || { echo "ERROR: cannot write $CF_UMBRELLA" >&2; exit 1; }
echo ">> CF_UMBRELLA=$CF_UMBRELLA"
echo ">> REPO=$REPO  (venv -> $REPO/.venv)"

# 1) uv (install into the umbrella if missing) -------------------------------
export UV_INSTALL_DIR="$CF_UMBRELLA/uv/bin"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$UV_INSTALL_DIR/uv" ]; then
  echo ">> installing uv -> $UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$UV_INSTALL_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# 2) venv + torch + transformers + deps (setup_uv_daic.sh honors CF_UMBRELLA) -
bash server/setup_uv_daic.sh

# 3) pre-download DreamCoder-7B to the now-local HF cache --------------------
source server/env_daic.sh
echo ">> pre-downloading DreamCoder-7B (~14GB) -> $HF_HOME (login node has internet) ..."
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Dream-org/Dream-Coder-v0-Instruct-7B")
print("  cached:", p)
PY

cat <<EOF

============================================================================
 FRESH ENV READY.  CF_UMBRELLA=$CF_UMBRELLA
 RUN the DreamCoder diffusion-step sweep (always export CF_UMBRELLA first):

   export CF_UMBRELLA=$CF_UMBRELLA
   cd $REPO && mkdir -p logs
   MODEL=DreamCoder-7B sbatch --export=ALL server/jobs/DiffStep-sweep.slurm
   squeue --me

 Results -> $REPO/results/diffusion_steps_benchmark/summary_*.csv
 (DiffusionGemma extra setup, only if you also need it here:
   CF_UMBRELLA=$CF_UMBRELLA PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh
  + a 96GB GPU. Heavier -- usually better to run dgemma on the main account.)
============================================================================
EOF
