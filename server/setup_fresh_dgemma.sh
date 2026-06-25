#!/bin/bash
# FRESH, self-contained env for running DiffusionGemma on a machine/user WITHOUT
# access to the original CoReFusion umbrella. Run ONCE on a LOGIN node (internet).
# HEAVY: builds a venv + pylibs_dgemma + downloads ~52GB of weights -> use tmux/screen.
#
#   CF_UMBRELLA=<a dir YOU can write to>  bash server/setup_fresh_dgemma.sh
#   e.g.  CF_UMBRELLA=/tudelft.net/staff-umbrella/NA  bash server/setup_fresh_dgemma.sh
#
# Builds: main venv (torch, used by dgemma) + pylibs_dgemma (transformers 5.11,
# the diffusion_gemma arch) + pre-downloads google/diffusiongemma-26B-A4B-it.
# To RUN you also need a 96GB GPU.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

: "${CF_UMBRELLA:?ERROR: set CF_UMBRELLA to a writable dir, e.g. CF_UMBRELLA=/tudelft.net/staff-umbrella/NA}"
export CF_UMBRELLA
mkdir -p "$CF_UMBRELLA" 2>/dev/null || { echo "ERROR: cannot write $CF_UMBRELLA" >&2; exit 1; }
echo ">> CF_UMBRELLA=$CF_UMBRELLA   REPO=$REPO   (venv -> $REPO/.venv)"

# 1) uv (install into the umbrella if missing) -------------------------------
export UV_INSTALL_DIR="$CF_UMBRELLA/uv/bin"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$UV_INSTALL_DIR/uv" ]; then
  echo ">> installing uv -> $UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$UV_INSTALL_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# 2) main venv (torch + base) -- dgemma uses torch from here, tfm 5.11 prepended at runtime
echo ">> building main venv (torch + transformers 4.57.1 + deps) ..."
bash server/setup_uv_daic.sh

# 3) pylibs_dgemma (transformers 5.11) + pre-download the ~52GB DiffusionGemma weights
echo ">> building pylibs_dgemma + pre-downloading DiffusionGemma-26B-A4B (~52GB) ..."
CF_UMBRELLA="$CF_UMBRELLA" PREDOWNLOAD=1 bash server/setup_dgemma_pylibs.sh

cat <<EOF

============================================================================
 DiffusionGemma env READY.  CF_UMBRELLA=$CF_UMBRELLA
 RUN (needs a 96GB GPU; ALWAYS export CF_UMBRELLA + use --export=ALL):

   export CF_UMBRELLA=$CF_UMBRELLA
   cd $REPO && mkdir -p logs
   sinfo -o "%N %G" | grep -iE "6000|pro"          # -> the <type> for the 96GB card
   sbatch --export=ALL --gres=gpu:<type>:1 server/jobs/DiffStep-dgemma.slurm

 Results -> $REPO/results/diffusion_steps_benchmark/summary_dgemma_*.csv
 The job dumps model.generation_config at startup; if the .out shows
 'max-steps field : NOT FOUND', send that dump back so we set the right field.
============================================================================
EOF
