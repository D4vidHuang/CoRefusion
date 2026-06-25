"""
Rebuild Figure 8 "RQ2 results" — two panels, unified CoReFusion blue/orange style.

Panel (a): grouped bar chart, Target-position Exact Match (%).
    Groups: RQ1 clean / All-masked / Target-only
    DiffuCoder-7B (BLUE):  31.1, 12.2, 3.1
    DreamCoder-7B (ORANGE):33.2, 13.9, 4.1
    (figures/new/analysis_data.json -> thesis.table5)

Panel (b): line plot, Mean per-sample EM (%) vs identifier-count buckets.
    Buckets: 1-10 / 11-20 / 21-50 / 51-100 / 100+
    DiffuCoder-7B (BLUE):  12.54, 10.22, 3.98, 0.87, 0.02
    DreamCoder-7B (ORANGE):13.86, 9.75, 4.21, 1.07, 0.0
    (figures/new/analysis_data.json -> deob_buckets * 100, all-masked experiment)
"""
import sys
sys.path.insert(0, "/Users/davidhuang/Desktop/CoRefusion/figures/rebuilt")

import json
import numpy as np
import matplotlib.pyplot as plt

from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE,
    ORANGE_DARK, ORANGE_MID, ORANGE_LIGHT, ORANGE_PALE,
    apply_style, blue_ramp, orange_ramp, savefig, lighten, CMAP_DIV)
apply_style()

DATA = "/Users/davidhuang/Desktop/CoRefusion/figures/new/analysis_data.json"
with open(DATA) as f:
    d = json.load(f)

# ---------------------------------------------------------------- thesis numbers
t5 = d["thesis"]["table5"]
# Panel (a) — target-position EM (%)
groups = ["RQ1\nclean", "All-masked", "Target-only"]
diffu_a = [t5["DiffuCoder-7B"]["rq1"], t5["DiffuCoder-7B"]["all_masked"], t5["DiffuCoder-7B"]["target_only"]]
dream_a = [t5["DreamCoder-7B"]["rq1"], t5["DreamCoder-7B"]["all_masked"], t5["DreamCoder-7B"]["target_only"]]
# expected: [31.1, 12.2, 3.1] and [33.2, 13.9, 4.1]

# Panel (b) — all-masked EM by identifier-count bucket (fractions -> %)
buckets = ["1-10", "11-20", "21-50", "51-100", "100+"]
db = d["deob_buckets"]
diffu_b = [db["DiffuCoder-7B"][b][0] * 100 for b in buckets]
dream_b = [db["DreamCoder-7B"][b][0] * 100 for b in buckets]
# expected: [12.54, 10.22, 3.98, 0.87, 0.02] and [13.86, 9.75, 4.21, 1.07, 0.0]

print("Panel (a) DiffuCoder:", diffu_a, " DreamCoder:", dream_a)
print("Panel (b) DiffuCoder:", [round(v, 2) for v in diffu_b],
      " DreamCoder:", [round(v, 2) for v in dream_b])

# ------------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))

# ---- Panel (a): grouped bars -------------------------------------------------
x = np.arange(len(groups))
w = 0.38
bars1 = axA.bar(x - w/2, diffu_a, w, label="DiffuCoder-7B", color=BLUE,
                edgecolor="white", linewidth=0.6, zorder=3)
bars2 = axA.bar(x + w/2, dream_a, w, label="DreamCoder-7B", color=ORANGE,
                edgecolor="white", linewidth=0.6, zorder=3)

axA.set_ylabel("Target-position Exact Match (%)")
axA.set_title("(a) Target-position EM")
axA.set_xticks(x)
axA.set_xticklabels(groups)
axA.set_ylim(0, 40)
axA.set_yticks(np.arange(0, 41, 5))

def label_bars(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)

label_bars(axA, bars1)
label_bars(axA, bars2)
axA.legend(loc="upper right")

# ---- Panel (b): line plot ----------------------------------------------------
xb = np.arange(len(buckets))
axB.plot(xb, diffu_b, marker="o", color=BLUE, label="DiffuCoder-7B",
         linewidth=2.2, markersize=6.5, markeredgecolor="white",
         markeredgewidth=0.8, zorder=3)
axB.plot(xb, dream_b, marker="s", color=ORANGE, label="DreamCoder-7B",
         linewidth=2.2, markersize=6.5, markeredgecolor="white",
         markeredgewidth=0.8, zorder=3)

# value labels next to each point: put the higher series' label above and the
# lower series' label below at each bucket so they never collide.
for xi, yd, yo in zip(xb, diffu_b, dream_b):
    if yd >= yo:
        axB.annotate("{:.1f}".format(yd), xy=(xi, yd), xytext=(0, 8),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=BLUE_DARK)
        axB.annotate("{:.1f}".format(yo), xy=(xi, yo), xytext=(0, -13),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=8.5, color=ORANGE_DARK)
    else:
        axB.annotate("{:.1f}".format(yo), xy=(xi, yo), xytext=(0, 8),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=ORANGE_DARK)
        axB.annotate("{:.1f}".format(yd), xy=(xi, yd), xytext=(0, -13),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=8.5, color=BLUE_DARK)

axB.set_ylabel("Mean per-sample EM (%)")
axB.set_xlabel("Number of distinct identifiers in S")
axB.set_title("(b) All-masked EM by identifier count")
axB.set_xticks(xb)
axB.set_xticklabels(buckets)
axB.set_ylim(-1.5, 16)
axB.set_yticks(np.arange(0, 16, 2.5))
axB.legend(loc="upper right")

fig.tight_layout(w_pad=2.2)
savefig(fig, "fig08_rq2_results")
