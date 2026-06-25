"""
Figure 13 — Commitment / unmasking-order analysis (DiffuCoder-7B).

(a) Distribution of the first denoising step at which a token first reaches
    confidence >= 0.8, for smell tokens vs context tokens (density-normalized
    overlaid histograms + mean lines).
(b) Distribution of the average flip step across all token occurrences,
    smell vs context.

Data : results/abc_exp/abc/unmasking_order_20260312_225458.csv
Cols : first_confident_step  (panel a)
       avg_flip_step         (panel b)
       is_smell_token        (smell vs context split)
"""
import sys
sys.path.insert(0, "/Users/davidhuang/Desktop/CoRefusion/figures/rebuilt")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE,
    ORANGE_DARK, ORANGE_MID, ORANGE_LIGHT, ORANGE_PALE,
    apply_style, blue_ramp, orange_ramp, savefig, lighten, CMAP_DIV)
apply_style()

DATA = "/Users/davidhuang/Desktop/CoRefusion/results/abc_exp/abc/unmasking_order_20260312_225458.csv"
TOTAL_STEPS = 64

df = pd.read_csv(DATA)
smell   = df[df["is_smell_token"] == True]
context = df[df["is_smell_token"] == False]
n_smell, n_ctx = len(smell), len(context)
print(f"smell={n_smell}  context={n_ctx}")

LBL_SMELL = f"Smell (n={n_smell})"
LBL_CTX   = f"Context (n={n_ctx:,})"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

# ---------------------------------------------------------------- panel (a)
col = "first_confident_step"
bins_a = np.linspace(0, TOTAL_STEPS, 33)   # 32 bins over 0..64

axA.hist(context[col], bins=bins_a, density=True, color=BLUE,
         alpha=0.55, edgecolor="white", linewidth=0.4, label=LBL_CTX, zorder=2)
axA.hist(smell[col], bins=bins_a, density=True, color=ORANGE,
         alpha=0.55, edgecolor="white", linewidth=0.4, label=LBL_SMELL, zorder=3)

m_ctx, m_smell = context[col].mean(), smell[col].mean()
axA.axvline(m_ctx,   color=BLUE_DARK,   linestyle="--", linewidth=1.8, zorder=4,
            label=f"Context mean = {m_ctx:.1f}")
axA.axvline(m_smell, color=ORANGE_DARK, linestyle="--", linewidth=1.8, zorder=5,
            label=f"Smell mean = {m_smell:.1f}")

axA.set_xlabel(r"First step at confidence $\geq$ 0.8")
axA.set_ylabel("Density")
axA.set_title("(a) First confident step", loc="left", fontweight="bold")
axA.set_xlim(0, TOTAL_STEPS)
axA.set_xticks(np.arange(0, TOTAL_STEPS + 1, 16))
axA.legend(loc="upper center", fontsize=8.5)

# ---------------------------------------------------------------- panel (b)
col = "avg_flip_step"
bins_b = np.linspace(0, TOTAL_STEPS, 33)

axB.hist(context[col], bins=bins_b, density=True, color=BLUE,
         alpha=0.55, edgecolor="white", linewidth=0.4, label=LBL_CTX, zorder=2)
axB.hist(smell[col], bins=bins_b, density=True, color=ORANGE,
         alpha=0.55, edgecolor="white", linewidth=0.4, label=LBL_SMELL, zorder=3)

mb_ctx, mb_smell = context[col].mean(), smell[col].mean()
axB.axvline(mb_ctx,   color=BLUE_DARK,   linestyle="--", linewidth=1.8, zorder=4,
            label=f"Context mean = {mb_ctx:.1f}")
axB.axvline(mb_smell, color=ORANGE_DARK, linestyle="--", linewidth=1.8, zorder=5,
            label=f"Smell mean = {mb_smell:.1f}")

axB.set_xlabel("Average flip step")
axB.set_ylabel("Density")
axB.set_title("(b) Average flip step", loc="left", fontweight="bold")
axB.set_xlim(0, TOTAL_STEPS)
axB.set_xticks(np.arange(0, TOTAL_STEPS + 1, 16))
axB.legend(loc="upper center", fontsize=8.5)

fig.tight_layout()
savefig(fig, "fig13_commitment_step")
