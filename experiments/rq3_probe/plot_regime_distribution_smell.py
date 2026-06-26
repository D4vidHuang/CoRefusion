#!/usr/bin/env python3
"""
Fig.10 (revised): regime slope chart using the DISTRIBUTION-based smell rank.

x = good (developer name r_gt)  vs  smell (median rank of smell names that the
model itself surfaces, by the objective rule in smell_rule.py — NOT an injected
vocabulary). Same three confidence regimes and same style as the original
fig_rq3_regime_inversion.

Reads the per-position CSV written by exp_distribution_smell_rank.py:
  columns: regime {OVERCONFIDENT,UNCERTAIN,CONFIDENT_RARE}, gt_rank, smell_med_rank
and medians per regime. (--smell-col smell_min_rank to plot the min instead.)

Usage:
  python plot_regime_distribution_smell.py <per_position.csv> [--fig-dir DIR] [--smell-col smell_med_rank]
  python plot_regime_distribution_smell.py --demo        # smoke-test with fake data
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

DISPLAY = {"OVERCONFIDENT": "HighConfident", "UNCERTAIN": "Uncertain",
           "CONFIDENT_RARE": "RareConfident"}
ORDER = ["HighConfident", "Uncertain", "RareConfident"]
COLORS = ["#3498db", "#9b59b6", "#e74c3c"]


def aggregate(csv_path, smell_col):
    """Accept either the per-position CSV (gt_rank, smell_med_rank, ...) or the
    already-aggregated *_summary.csv (median_gt_rank, median_smell_med_rank)."""
    rows = list(csv.DictReader(open(csv_path)))
    if rows and "median_gt_rank" in rows[0]:                       # summary format
        scol = "median_" + smell_col
        out = {}
        for r in rows:
            out[DISPLAY.get(r["regime"], r["regime"])] = (
                float(r["median_gt_rank"]), float(r[scol]))
        return [(k, out[k][0], out[k][1]) for k in ORDER if k in out]
    agg = {}                                                        # per-position format
    for r in rows:
        disp = DISPLAY.get(r["regime"], r["regime"])
        agg.setdefault(disp, []).append((float(r["gt_rank"]), float(r[smell_col])))
    return [(k, float(np.median([a for a, _ in agg[k]])),
            float(np.median([b for _, b in agg[k]]))) for k in ORDER if k in agg]


def make_demo_csv(path):
    import random
    rng = random.Random(0)
    bands = {"OVERCONFIDENT": (80, 320), "UNCERTAIN": (520, 540),
             "CONFIDENT_RARE": (11000, 740)}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "gt_name", "gt_rank", "regime",
                    "smell_med_rank", "smell_min_rank", "n_smell", "target_idx"])
        for rg, (g, s) in bands.items():
            for i in range(200):
                w.writerow([i, "x", int(g * rng.uniform(.5, 2)), rg,
                            int(s * rng.uniform(.7, 1.4)), int(s * .2), 900, 5])


def plot(regimes, fig_dir, stem):
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    L_OFF = [(-4, 2), (-4, 2), (-4, 5)]
    R_OFF = [(7, -4), (7, 5), (7, 13)]
    for i, (name, rgt, rsm) in enumerate(regimes):
        ax.plot([0, 1], [rgt, rsm], "-o", lw=1.4, ms=4, color=COLORS[i % 3], label=name)
        ax.annotate("%d" % rgt, (0, rgt), textcoords="offset points",
                    xytext=L_OFF[i % 3], ha="right", fontsize=7)
        ax.annotate("%d" % rsm, (1, rsm), textcoords="offset points",
                    xytext=R_OFF[i % 3], ha="left", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlim(-0.20, 1.18)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["good\n(developer)", "smell\n(distribution)"])
    ax.set_ylabel("Median rank in output distribution (log)")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="per-position CSV from exp_distribution_smell_rank.py")
    ap.add_argument("--fig-dir", default="figures/new/rq3")
    ap.add_argument("--smell-col", default="smell_med_rank",
                    choices=["smell_med_rank", "smell_min_rank"])
    ap.add_argument("--stem", default="fig_rq3_regime_inversion_dist")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        tmp = "/tmp/_smell_demo.csv"
        make_demo_csv(tmp)
        plot(aggregate(tmp, a.smell_col), a.fig_dir, a.stem + "_DEMO")
    else:
        if not a.csv:
            ap.error("provide the per-position CSV, or use --demo")
        plot(aggregate(a.csv, a.smell_col), a.fig_dir, a.stem)
