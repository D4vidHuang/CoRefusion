"""
Figure 7 — Diffusion step sensitivity of DiffuCoder-7B-Base on the RefineID test set.
Two panels:
  (a) EM (%) vs diffusion steps T (log2 x-axis), with reference line + annotations.
  (b) per-sample latency (s) vs T on a log-log scale (near-linear).

Single model -> BLUE line+markers throughout. Data are from Table IV.
"""
import sys
sys.path.insert(0, "/Users/davidhuang/Desktop/CoRefusion/figures/rebuilt")
from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE,
    ORANGE_DARK, ORANGE_MID, ORANGE_LIGHT, ORANGE_PALE,
    apply_style, blue_ramp, orange_ramp, savefig, lighten, CMAP_DIV)
apply_style()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# ------------------------------------------------------------------ Table IV
T    = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
EM   = np.array([26.20, 30.10, 30.30, 30.50, 30.60, 30.80, 31.00])
TIME = np.array([0.313, 0.618, 1.238, 2.477, 4.957, 9.916, 19.831])

REF_EM = 31.0  # reference value at T=64

# ------------------------------------------------------------------ figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

# ============================================================== Panel (a) EM
axL.axhline(REF_EM, color=GRAY, ls="--", lw=1.3, zorder=1)
axL.plot(T, EM, color=BLUE, marker="o", ms=6.5, lw=2.2, zorder=3,
         label="DiffuCoder-7B-Base")

# reference-line label
axL.text(64, REF_EM + 0.12, "Reference (T=64)", color=GRAY,
         ha="right", va="bottom", fontsize=9)

# annotate first point (the dip)
axL.annotate("26.2% (step=1)",
             xy=(1, 26.20), xytext=(1.7, 27.4),
             color=BLUE_DARK, fontsize=9, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=BLUE_DARK, lw=1.0,
                             connectionstyle="arc3,rad=-0.2"))

# plateau / speedup notes (orange)
axL.text(8, 29.55, "~0.9 pp plateau", color=ORANGE_DARK,
         fontsize=9.5, ha="center", va="top", style="italic")
axL.text(8, 28.95, "32x speedup at T=2 vs T=64", color=ORANGE,
         fontsize=8.8, ha="center", va="top")

axL.set_xscale("log", base=2)
axL.set_xlim(0.85, 80)
axL.set_ylim(25, 32)
axL.set_xticks(T)
axL.xaxis.set_major_formatter(FixedFormatter([str(int(t)) for t in T]))
axL.xaxis.set_minor_locator(NullLocator())
axL.set_xlabel("Diffusion steps $T$ (log scale)")
axL.set_ylabel("Exact Match (%)")
axL.set_title("(a) Accuracy vs. diffusion steps", fontsize=11, pad=8)
axL.legend(loc="lower right", frameon=False)
axL.grid(True, axis="y", color=GRID, lw=0.9)
axL.set_axisbelow(True)

# ============================================================== Panel (b) time
axR.plot(T, TIME, color=BLUE, marker="o", ms=6.5, lw=2.2, zorder=3,
         label="DiffuCoder-7B-Base")

axR.set_xscale("log", base=2)
axR.set_yscale("log", base=10)
axR.set_xlim(0.85, 80)
axR.set_xticks(T)
axR.xaxis.set_major_formatter(FixedFormatter([str(int(t)) for t in T]))
axR.xaxis.set_minor_locator(NullLocator())

yticks = [0.2, 0.5, 1, 2, 5, 10, 20]
axR.set_ylim(0.25, 25)
axR.yaxis.set_major_locator(FixedLocator(yticks))
axR.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yticks]))
axR.yaxis.set_minor_locator(NullLocator())

axR.set_xlabel("Diffusion steps $T$ (log scale)")
axR.set_ylabel("Time per sample (s)")
axR.set_title("(b) Per-sample latency vs. diffusion steps", fontsize=11, pad=8)
axR.legend(loc="upper left", frameon=False)
axR.grid(True, which="major", axis="both", color=GRID, lw=0.9)
axR.set_axisbelow(True)

fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.92, wspace=0.24)

savefig(fig, "fig07_diffusion_steps")
