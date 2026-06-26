"""Fig 6 (paper) / fig07 — Diffusion step sensitivity on the RefineID test set.

DATA-DRIVEN: reads the latest step-sweep summary written by
    experiments/exp_diffusion_steps_benchmark.py
    -> results/diffusion_steps_benchmark/summary_<ts>.csv
and plots one line per model present (EM vs steps, and per-sample latency vs
steps). If no summary is found it falls back to the published DiffuCoder-7B-Base
numbers so the figure still builds.

NOTE: the step sweep is only well-defined for the FIXED-CANVAS dLLMs
(DiffuCoder-7B, DreamCoder-7B), which expose ``diffusion_generate(steps=T)``.
DreamOn-7B (variable canvas, transfers tokens until convergence) and
DiffusionGemma-26B-A4B (block-AR, internal entropy-bounded sampler) do not
expose the same free step knob, so they are not part of this figure.
"""
import os
import sys
import csv
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID,
    BLUE_DARK, ORANGE_DARK, apply_style, savefig)
apply_style()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUMMARY_GLOB = os.path.join(REPO, "results", "diffusion_steps_benchmark", "summary_*.csv")

CYAN = "#00A6D6"
COLOR = {"DiffuCoder-7B": BLUE, "DiffuCoder-7B-Base": BLUE,
         "DreamCoder-7B": ORANGE, "DreamCoder-7B-Base": ORANGE,
         "DiffusionGemma-26B-A4B": CYAN}
LABEL = {"DiffuCoder-7B": "DiffuCoder-7B-Base", "DreamCoder-7B": "DreamCoder-7B",
         "DiffusionGemma-26B-A4B": "DiffusionGemma-26B (no early stop)"}
ORDER = ["DiffuCoder-7B", "DiffuCoder-7B-Base", "DreamCoder-7B", "DiffusionGemma-26B-A4B"]

# ── published DiffuCoder-7B-Base curve (fallback / authoritative) ────────────
PUBLISHED = {"DiffuCoder-7B-Base": {
    "steps": [1, 2, 4, 8, 16, 32, 64],
    "em":    [26.20, 30.10, 30.30, 30.50, 30.60, 30.80, 31.00],   # %
    "time":  [0.313, 0.618, 1.238, 2.477, 4.957, 9.916, 19.831],  # s/sample
}}


def load_series():
    """model -> {steps, em(%), time(s)}, merged across ALL summary CSVs (newest
    file wins per model; covers separate DiffuCoder/DreamCoder and DiffusionGemma
    runs). summary_dgemma_*.csv also matches the glob."""
    series = {}
    for fpath in sorted(glob.glob(SUMMARY_GLOB)):   # oldest -> newest by name
        by = {}
        for r in csv.DictReader(open(fpath, encoding="utf-8")):
            m = r["model"]
            by.setdefault(m, []).append((int(r["steps"]),
                                         float(r["exact_match_rate"]) * 100,
                                         float(r["mean_time_per_sample"])))
        for m, rows in by.items():
            rows.sort()
            series[m] = {"steps": [s for s, _, _ in rows],
                         "em": [e for _, e, _ in rows],
                         "time": [t for _, _, t in rows]}   # later file overrides
    if series:
        print(f"loaded {len(series)} model(s) from summary CSVs: {list(series)}")
    # ensure DiffuCoder is shown even if it was not re-run this round
    if not any("DiffuCoder" in m for m in series):
        series.update(PUBLISHED)
        print("no DiffuCoder run found -> using published DiffuCoder-7B-Base curve")
    return series


series = load_series()
models = [m for m in ORDER if m in series] + [m for m in series if m not in ORDER]

# ── figure ──────────────────────────────────────────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

all_steps = sorted({s for m in models for s in series[m]["steps"]})
xt = np.array(all_steps, dtype=float)

for i, m in enumerate(models):
    col = COLOR.get(m, CYAN)
    s = np.array(series[m]["steps"], float)
    em = np.array(series[m]["em"], float)
    tm = np.array(series[m]["time"], float)
    lab = LABEL.get(m, m)
    axL.plot(s, em, color=col, marker="o", ms=6.0, lw=2.2, zorder=3, label=lab)
    # faint plateau reference at this model's largest-step EM
    axL.axhline(em[np.argmax(s)], color=col, ls="--", lw=1.0, alpha=0.4, zorder=1)
    axR.plot(s, tm, color=col, marker="o", ms=6.0, lw=2.2, zorder=3, label=lab)

# ── panel (a) EM ────────────────────────────────────────────────────────────
axL.set_xscale("log", base=2)
axL.set_xlim(0.85, max(xt) * 1.25)
ymin = min(min(series[m]["em"]) for m in models)
ymax = max(max(series[m]["em"]) for m in models)
axL.set_ylim(np.floor(ymin) - 1, np.ceil(ymax) + 1)
axL.set_xticks(xt)
axL.xaxis.set_major_formatter(FixedFormatter([str(int(t)) for t in xt]))
axL.xaxis.set_minor_locator(NullLocator())
axL.set_xlabel("Diffusion steps $T$ (log scale)")
axL.set_ylabel("Exact Match (%)")
axL.set_title("(a)", fontsize=11, pad=8)
axL.legend(loc="lower right", frameon=False)
axL.grid(True, axis="y", color=GRID, lw=0.9)
axL.set_axisbelow(True)

# ── panel (b) latency ───────────────────────────────────────────────────────
axR.set_xscale("log", base=2)
axR.set_yscale("log", base=10)
axR.set_xlim(0.85, max(xt) * 1.25)
axR.set_xticks(xt)
axR.xaxis.set_major_formatter(FixedFormatter([str(int(t)) for t in xt]))
axR.xaxis.set_minor_locator(NullLocator())
yticks = [0.2, 0.5, 1, 2, 5, 10, 20]
axR.set_ylim(0.2, 30)
axR.yaxis.set_major_locator(FixedLocator(yticks))
axR.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yticks]))
axR.yaxis.set_minor_locator(NullLocator())
axR.set_xlabel("Diffusion steps $T$ (log scale)")
axR.set_ylabel("Time per sample (s)")
axR.set_title("(b)", fontsize=11, pad=8)
axR.legend(loc="upper left", frameon=False)
axR.grid(True, which="major", axis="both", color=GRID, lw=0.9)
axR.set_axisbelow(True)

fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.92, wspace=0.24)
savefig(fig, "fig07_diffusion_steps")
plt.close(fig)
