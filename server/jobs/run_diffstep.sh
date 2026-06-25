#!/bin/bash
# Launch the Fig 6 diffusion-step sweeps (DreamCoder + DiffusionGemma) from THIS
# repo, wherever it is cloned (e.g. /tudelft.net/staff-umbrella/NA/CoReFusion).
#
#   cd /tudelft.net/staff-umbrella/NA/CoReFusion
#   bash server/jobs/run_diffstep.sh
#   DGEMMA_GRES=<type> bash server/jobs/run_diffstep.sh   # if the 96GB auto-detect misses
#
# It (1) verifies the env (PROJECT_DIR + venv) so a broken NA setup fails fast
# BEFORE submitting an 11h job, (2) submits the DreamCoder steps sweep (full),
# (3) submits the DiffusionGemma steps sweep as a SMOKE first (it must dump
# generation_config so we can confirm the max-steps / early-stop field names).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

echo "== verifying env =="
source server/env_daic.sh
echo "  PROJECT_DIR = $PROJECT_DIR"
echo "  UMBRELLA    = $UMBRELLA   (caches/venv/pylibs reused from here)"
python -c "import torch,transformers; print('  torch',torch.__version__,'| tfm',transformers.__version__,'| cuda',torch.cuda.is_available())" \
  || { echo "  [FATAL] venv not usable. Fix first (CF_UMBRELLA=... or bash server/setup_uv_daic.sh)." >&2; exit 1; }
case "$PROJECT_DIR" in
  *"$(basename "$REPO")") : ;;
  *) echo "  [warn] PROJECT_DIR != this repo ($REPO) -- jobs may run elsewhere." >&2 ;;
esac

echo
echo "== 1) DreamCoder-7B diffusion-step sweep (fixed-canvas, full) =="
MODEL=DreamCoder-7B sbatch server/jobs/DiffStep-sweep.slurm

echo
echo "== 2) DiffusionGemma-26B-A4B diffusion-step sweep (96GB card + pylibs_dgemma) =="
GRES="${DGEMMA_GRES:-}"
if [ -z "$GRES" ]; then
  GRES=$(sinfo -h -o '%G' 2>/dev/null | tr ',' '\n' \
         | grep -ioE 'gpu:[a-z0-9_]*6000[a-z0-9_]*' | head -1 | sed 's/^gpu://') || true
fi
if [ -n "$GRES" ]; then
  echo "  gres=gpu:${GRES}:1 ; submitting SMOKE (--debug, 5 samples) to dump generation_config ..."
  sbatch --gres=gpu:"${GRES}":1 server/jobs/DiffStep-dgemma.slurm --max-samples 5 --debug
  echo "  -> check logs/diffstep-dgemma-*.out: look for 'max-steps field : ...' (not NOT FOUND)."
  echo "  -> then FULL:  sbatch --gres=gpu:${GRES}:1 server/jobs/DiffStep-dgemma.slurm"
else
  echo "  [!] could not auto-detect a 96GB GPU GRES. Find it:"
  echo "        sinfo -o \"%N %G\" | grep -iE \"6000|pro\""
  echo "      then re-run:  DGEMMA_GRES=<type> bash server/jobs/run_diffstep.sh"
fi

echo
squeue --me 2>/dev/null || true
