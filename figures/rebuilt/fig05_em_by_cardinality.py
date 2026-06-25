"""Fig 4 (paper) / fig05 — All-sites Exact Match vs rename-set cardinality |S|.

DATA-DRIVEN: every number is recomputed from the unified RQ1 predictions by
    python analysis/em_by_cardinality.py        -> results/unified_refineID/em_by_cardinality.csv
so the figure always reflects the authoritative leaderboard (em_gated), and new
models (DreamOn, DiffusionGemma) appear automatically.

Design — "all models" without clutter:
  * The four diffusion LLMs are drawn as distinct bold lines (orange ramp).
  * The Decoder-only (FIM-AR) and Encoder-decoder (Seq2Seq) families are each
    drawn as a min--max band + mean line, so every benchmarked model is
    represented while the dLLM/AR contrast stays legible.

Rebuilt with the unified CoReFusion blue/orange style.
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID, LIGHT_GRAY,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE,
    ORANGE_DARK, ORANGE_MID, ORANGE_LIGHT, ORANGE_PALE,
    apply_style, orange_ramp, savefig)
apply_style()

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(REPO, "results", "unified_refineID", "em_by_cardinality.csv")

# x buckets (must match analysis/em_by_cardinality.py BUCKETS)
BIN_LABELS = ["|S|=1", "|S|=2", "|S|=3-5", "|S|=6-10", "|S|>=11"]
PRETTY = {"|S|=1": "1", "|S|=2": "2", "|S|=3-5": "3-5",
          "|S|=6-10": "6-10", "|S|>=11": r"$\geq$11"}

# The diffusion LLMs to draw as distinct lines, dark -> light (high EM -> low).
DLLM_ORDER = ["DreamCoder-7B", "DiffuCoder-7B", "DiffusionGemma-26B-A4B", "DreamOn-7B"]
DLLM_LABEL = {"DiffusionGemma-26B-A4B": "DiffusionGemma-26B"}   # short legend label
DLLM_MARKERS = ["o", "s", "D", "^"]

# ── load ──────────────────────────────────────────────────────────────────
if not os.path.exists(CSV):
    sys.exit(f"missing {CSV}\n  run: python analysis/em_by_cardinality.py")

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
def series(model):
    by = {r["bucket"]: r for r in rows if r["model"] == model}
    return [float(by[b]["em_pct"]) if b in by and by[b]["em_pct"] != "" else np.nan
            for b in BIN_LABELS]

fam_models = {}
for r in rows:
    fam_models.setdefault(r["family"], set()).add(r["model"])

def family_matrix(family, exclude=()):
    ms = sorted(m for m in fam_models.get(family, ()) if m not in exclude)
    M = np.array([series(m) for m in ms]) if ms else np.empty((0, len(BIN_LABELS)))
    return ms, M

x = np.arange(len(BIN_LABELS))

# ── figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.2))

# crossover band over the "|S|=3-5" bucket
band_idx = BIN_LABELS.index("|S|=3-5")
ax.axvspan(band_idx - 0.5, band_idx + 0.5, color=ORANGE_PALE, alpha=0.30, zorder=0)
ax.annotate("dLLM / AR\ncrossover", xy=(band_idx, 44), ha="center", va="center",
            fontsize=8.0, style="italic", color=GRAY, zorder=1)

# ── family bands (every non-dLLM model is represented here) ────────────────
def draw_band(family, color, color_pale, label):
    ms, M = family_matrix(family)
    if M.shape[0] == 0:
        return None
    lo = np.nanmin(M, axis=0); hi = np.nanmax(M, axis=0); mean = np.nanmean(M, axis=0)
    ax.fill_between(x, lo, hi, color=color_pale, alpha=0.55, zorder=2, linewidth=0)
    h, = ax.plot(x, mean, color=color, lw=1.8, ls="--", marker="o", ms=4.5,
                 mfc=color, mec="white", mew=0.5, zorder=3,
                 label=f"{label} (n={len(ms)}, mean)")
    return h

h_ar  = draw_band("Decoder-only",     BLUE, BLUE_PALE, "FIM-AR")
h_s2s = draw_band("Encoder-decoder",  GRAY, "#E9ECEF", "Seq2Seq")

# ── dLLM distinct lines (on top) ──────────────────────────────────────────
present_dllm = [m for m in DLLM_ORDER if m in fam_models.get("dLLM", ())]
oramp = orange_ramp(max(len(present_dllm), 2))
dllm_handles = []
for (model, col, mk) in zip(present_dllm, oramp, DLLM_MARKERS):
    ys = series(model)
    h, = ax.plot(x, ys, color=col, marker=mk, mfc=col, mec="white", mew=0.7,
                 ms=7.0, lw=2.6, zorder=6, label=DLLM_LABEL.get(model, model))
    dllm_handles.append(h)

# ── axes ──────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels([PRETTY[b] for b in BIN_LABELS])
ax.set_xlim(-0.4, len(BIN_LABELS) - 0.6)
ax.set_xlabel("Number of masked sites")
ax.set_ylabel("All-sites Exact Match (%)")
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))

# ── grouped legend: two columns, each with its heading on the TOP row ──────
# matplotlib fills legends column-major, so we pass column 1 then column 2 and
# pad the shorter column with blank entries to keep both headings aligned at top.
HEAD_D = "Diffusion LLMs"
HEAD_F = "Autoregressive / Seq2Seq"
header_d = Line2D([], [], color="none", label=HEAD_D)
header_f = Line2D([], [], color="none", label=HEAD_F)
fam_handles = [h for h in (h_ar, h_s2s) if h is not None]

col1 = [header_d] + dllm_handles          # heading + diffusion models
col2 = [header_f] + fam_handles           # heading + AR/Seq2Seq family means
nrow = max(len(col1), len(col2))
def _blank():
    return Line2D([], [], color="none", label=" ")
col1 += [_blank() for _ in range(nrow - len(col1))]
col2 += [_blank() for _ in range(nrow - len(col2))]

handles = col1 + col2                      # column-major: col1 -> left, col2 -> right
labels = [h.get_label() for h in handles]
leg = ax.legend(handles, labels, loc="upper right", ncol=2,
                handlelength=1.8, columnspacing=1.3, labelspacing=0.45,
                fontsize=8.0, borderaxespad=0.4,
                frameon=True, framealpha=0.93, edgecolor=GRID)
for txt in leg.get_texts():
    if txt.get_text() in (HEAD_D, HEAD_F):
        txt.set_fontweight("bold"); txt.set_color(INK)

fig.tight_layout()
savefig(fig, "fig05_em_by_cardinality")
plt.close(fig)
