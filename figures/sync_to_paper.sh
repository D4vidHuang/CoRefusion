#!/bin/bash
# Copy the rebuilt / reproduced figures into the paper repo (sibling dir).
# Run after rebuilding any figure so the paper picks up the new PDF/PNG.
#
#   bash figures/sync_to_paper.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"            # CoRefusion repo root
PAPER="${PAPER_DIR:-$HOME/Desktop/CoReFusion_ICSE27}"
DEST="$PAPER/figures"

if [ ! -d "$DEST" ]; then
  echo "FATAL: paper figures dir not found: $DEST" >&2
  echo "  set PAPER_DIR=/path/to/CoReFusion_ICSE27 and re-run." >&2
  exit 1
fi

# (source path, basename used in the paper)
copy() {  # $1 = source file, $2 = dest basename
  if [ -f "$1" ]; then
    cp -f "$1" "$DEST/$2"
    echo "  ok   $2"
  else
    echo "  MISS $2   (source not built yet: $1)"
  fi
}

echo "Syncing figures -> $DEST"
# Fig 2 (paper): RQ2 obfuscation example ("rename set", not "smell set")
copy "$HERE/figures/rebuilt/fig04_rq2_example.pdf" "fig04_rq2_example.pdf"
# Fig 4 (paper): all-sites EM by cardinality, all models (data-driven)
copy "$HERE/figures/rebuilt/fig05_em_by_cardinality.pdf" "fig05_em_by_cardinality.pdf"
# Fig 5 (paper): mask-token count ablation (+ DreamOn once its run lands)
copy "$HERE/figures/rebuilt/fig06_mask_count_ablation.pdf" "fig06_mask_count_ablation.pdf"
# Fig 8 (paper): RQ2 results (+ DreamOn once its deobfuscation run lands)
copy "$HERE/figures/new/fig8_rq2_em_june.pdf" "fig8_rq2_em_june.pdf"
echo "done."
