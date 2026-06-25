"""
Fig 12 — Distribution of mean entropy change (Delta-H) across the denoising
trajectory for smell tokens (n=158) vs context tokens (n=19,838).

Overlaid density histograms + KDE, blue=context, orange=smell, with vertical
dashed lines at each group's mean.

Data: results/abc_exp/abc/token_ranking_20260312_230225.csv
  column  mean_entropy_change  -> per-token mean entropy change across denoising
  flag    is_smell_token       -> True = smell token, False = context token
"""
import sys
sys.path.insert(0, "/Users/davidhuang/Desktop/CoRefusion/figures/rebuilt")
from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE,
    ORANGE_DARK, ORANGE_MID, ORANGE_LIGHT, ORANGE_PALE,
    apply_style, blue_ramp, orange_ramp, savefig, lighten, CMAP_DIV)
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

CSV = "/Users/davidhuang/Desktop/CoRefusion/results/abc_exp/abc/token_ranking_20260312_230225.csv"
df = pd.read_csv(CSV)

smell   = df.loc[df["is_smell_token"] == True,  "mean_entropy_change"].to_numpy()
context = df.loc[df["is_smell_token"] == False, "mean_entropy_change"].to_numpy()
n_smell, n_context = len(smell), len(context)
print(f"smell n={n_smell}  context n={n_context}")

# clip x range for visibility (context has a long thin tail up to ~8)
XMAX = 1.6
bins = np.linspace(0.0, XMAX, 41)

fig, ax = plt.subplots(figsize=(7.2, 4.3))

# --- overlaid density histograms -----------------------------------------
ax.hist(context, bins=bins, density=True, color=BLUE, alpha=0.42,
        edgecolor=BLUE_DARK, linewidth=0.4, label=f"Context (n={n_context:,})", zorder=2)
ax.hist(smell, bins=bins, density=True, color=ORANGE, alpha=0.50,
        edgecolor=ORANGE_DARK, linewidth=0.4, label=f"Smell (n={n_smell})", zorder=3)

# --- smooth KDE overlays --------------------------------------------------
xs = np.linspace(0.0, XMAX, 400)
kde_ctx   = gaussian_kde(context)
kde_smell = gaussian_kde(smell)
ax.plot(xs, kde_ctx(xs),   color=BLUE_DARK,   lw=1.8, zorder=4)
ax.plot(xs, kde_smell(xs), color=ORANGE_DARK, lw=1.8, zorder=5)

# --- mean lines -----------------------------------------------------------
m_ctx, m_smell = context.mean(), smell.mean()
ymax = ax.get_ylim()[1]
ax.axvline(m_ctx,   color=BLUE_DARK,   ls="--", lw=1.3, alpha=0.9, zorder=6)
ax.axvline(m_smell, color=ORANGE_DARK, ls="--", lw=1.3, alpha=0.9, zorder=6)
ax.annotate(f"mean = {m_ctx:.2f}", xy=(m_ctx, ymax * 0.93),
            xytext=(m_ctx - 0.02, ymax * 0.93), ha="right", va="top",
            color=BLUE_DARK, fontsize=9)
ax.annotate(f"mean = {m_smell:.2f}", xy=(m_smell, ymax * 0.78),
            xytext=(m_smell + 0.03, ymax * 0.78), ha="left", va="top",
            color=ORANGE_DARK, fontsize=9)

# --- cosmetics ------------------------------------------------------------
ax.set_xlabel(r"Mean entropy change ($\Delta H$)")
ax.set_ylabel("Density")
ax.set_xlim(0.0, XMAX)
ax.set_ylim(0, ymax)
ax.legend(loc="upper right", frameon=False)

savefig(fig, "fig12_entropy_change")
