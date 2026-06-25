"""
Fig 10 — Cosine similarity between smelly and clean hidden states at the target
position (averaged over 926 RefineID samples).

Two panels:
  (a) mean cosine similarity vs transformer layer (0..28, 29 layers)
  (b) mean cosine similarity vs denoising step  (0..31, 32 steps)

Single BLUE line+markers per panel, a GRAY dashed overall-mean line, and the
final third of each x-axis shaded ORANGE_PALE to mark where the divergence
concentrates.
"""
import sys
sys.path.insert(0, "/Users/davidhuang/Desktop/CoRefusion/figures/rebuilt")
from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE,
    ORANGE_DARK, ORANGE_MID, ORANGE_LIGHT, ORANGE_PALE,
    apply_style, blue_ramp, orange_ramp, savefig, lighten, CMAP_DIV)
apply_style()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LAYER_CSV = "/Users/davidhuang/Desktop/CoRefusion/results/results_20260405_2056/edit_signals_layer_wise.csv"
STEP_CSV  = "/Users/davidhuang/Desktop/CoRefusion/results/results_20260405_2056/edit_signals_diffusion_steps.csv"


def num(val):
    return int(str(val).split("_")[-1])


def summarize(path):
    df = pd.read_csv(path)
    g = df.groupby("step")["cosine_sim"].mean().reset_index()
    g["x"] = g["step"].apply(num)
    g = g.sort_values("x")
    return g["x"].to_numpy(), g["cosine_sim"].to_numpy()


lx, ly = summarize(LAYER_CSV)
sx, sy = summarize(STEP_CSV)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.3))


def panel(ax, x, y, xlabel, title):
    xmax = x.max()
    # shade the final third of the x-axis (where divergence concentrates)
    third_start = x.min() + 2.0 / 3.0 * (xmax - x.min())
    ax.axvspan(third_start, xmax, color=ORANGE_PALE, alpha=0.55, lw=0, zorder=0)

    overall = float(np.mean(y))
    ax.axhline(overall, color=GRAY, ls="--", lw=1.4, zorder=2,
               label=f"overall mean = {overall:.3f}")

    ax.plot(x, y, color=BLUE, lw=2.2, marker="o", markersize=5.0,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3, label="mean cosine similarity")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.set_xlim(x.min() - 0.6, xmax + 0.6)
    ax.legend(loc="upper left", fontsize=9)
    return overall


panel(axL, lx, ly, "Transformer layer", "(a) Across transformer layers")
panel(axR, sx, sy, "Denoising step",     "(b) Across denoising steps")

fig.tight_layout()
savefig(fig, "fig10_cosine_similarity")
print("layer overall mean", float(np.mean(ly)), "step overall mean", float(np.mean(sy)))
