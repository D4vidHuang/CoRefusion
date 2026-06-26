"""
Fig 9 — Context sensitivity under progressive context masking on 200 RefineID snippets.

Two panels:
  (a) Mean entropy H(alpha) at the target (ground-truth) position vs masked-context
      fraction alpha, one line per confidence regime.
  (b) Delta-H (H at high alpha minus H at alpha=0) per regime.

Regime -> display label -> colour:
  OVERCONFIDENT  -> HighConfident -> BLUE
  UNCERTAIN      -> Uncertain     -> GRAY
  CONFIDENT_RARE -> RareConfident -> ORANGE

Data: results/overconfidence_localization/DiffuCoder-7B-Base_20260303_100019.csv
Entropy at the target position = rows with probe_type == 'gt' (the ground-truth
identifier slot), column 'entropy', averaged over snippets at each alpha.
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

CSV = "/Users/davidhuang/Desktop/CoRefusion/results/overconfidence_localization/DiffuCoder-7B-Base_20260303_100019.csv"

# ----------------------------------------------------------------- load + reduce
df = pd.read_csv(CSV)
gt = df[df["probe_type"] == "gt"].copy()          # entropy at the target slot

# regime -> (display label, colour, plot order)
REGIMES = [
    ("OVERCONFIDENT",  "HighConfident", BLUE),
    ("UNCERTAIN",      "Uncertain",     GRAY),
    ("CONFIDENT_RARE", "RareConfident", ORANGE),
]

# x range: full masking sweep up to alpha=0.8 (the regime-stratified window)
ALPHA_MAX = 0.8
alphas = sorted(a for a in gt["alpha"].unique() if a <= ALPHA_MAX + 1e-9)

# mean entropy H(alpha) per regime
mean_H = (gt.groupby(["regime", "alpha"])["entropy"].mean().unstack("alpha"))
sem_H  = (gt.groupby(["regime", "alpha"])["entropy"]
            .agg(lambda s: s.std(ddof=1) / np.sqrt(len(s))).unstack("alpha"))

n_snips = gt[gt["alpha"] == 0.0]["sample_id"].nunique()  # 200

# Delta-H = H(alpha_max) - H(alpha=0)
deltaH = {reg: mean_H.loc[reg, ALPHA_MAX] - mean_H.loc[reg, 0.0]
          for reg, _, _ in REGIMES}

# ------------------------------------------------------------------- the figure
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(10.2, 4.0),
    gridspec_kw=dict(width_ratios=[1.65, 1.0], wspace=0.30))

# ---- panel (a): H(alpha) lines -------------------------------------------------
for reg, lab, col in REGIMES:
    y   = mean_H.loc[reg, alphas].values
    err = sem_H.loc[reg, alphas].values
    axA.fill_between(alphas, y - err, y + err, color=col, alpha=0.13, lw=0)
    axA.plot(alphas, y, color=col, marker="o", markersize=5.5,
             linewidth=2.4, label=lab, zorder=3,
             markeredgecolor="white", markeredgewidth=0.8)

axA.set_xlabel(r"Masked-context fraction $\alpha$")
axA.set_ylabel(r"Mean entropy $H$")
axA.set_xlim(-0.03, ALPHA_MAX + 0.03)
axA.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
axA.margins(y=0.08)
axA.legend(title="Confidence regime", title_fontsize=9,
           loc="upper left", frameon=False)
axA.set_title("(a)", fontsize=10, pad=8)

# ---- panel (b): Delta-H bars ---------------------------------------------------
labels = [lab for _, lab, _ in REGIMES]
cols   = [col for _, _, col in REGIMES]
vals   = [deltaH[reg] for reg, _, _ in REGIMES]
xpos   = np.arange(len(labels))

bars = axB.bar(xpos, vals, width=0.62, color=cols,
               edgecolor="white", linewidth=0.8, zorder=3)
for x, v in zip(xpos, vals):
    axB.text(x, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
             fontsize=9, color=INK)

axB.set_xticks(xpos)
axB.set_xticklabels(labels, rotation=12, ha="right")
axB.set_ylabel(r"$\Delta H$  ($\alpha{=}%.1f$ $-$ $\alpha{=}0$)" % ALPHA_MAX)
axB.set_ylim(0, max(vals) * 1.18)
axB.set_title("(b)", fontsize=10, pad=8)

savefig(fig, "fig09_context_sensitivity")
print("Delta-H:", {k: round(v, 3) for k, v in deltaH.items()})
print("n snippets:", n_snips)
