"""Fig 6 — Mask-token count ablation, two panels side by side.

(a) Grouped bar chart: EM (%) vs mask tokens per identifier k in {1..5}.
    DiffuCoder-7B (blue) and DreamCoder-7B (orange) are FIXED-CANVAS dLLMs and
    peak at k=2 (published numbers, raw data under results/1t5t_exp/).
    DreamOn-7B (cyan) is a VARIABLE-CANVAS dLLM; its k-sweep is recomputed by
        experiments/1t5t_exp/part2_dreamon_mask_ablation.py
        -> results/1t5t_exp/dreamon_mask_ablation_summary.csv
    and is added automatically once that run lands (graceful if absent).
(b) GT identifier length distribution (% of samples) by sub-word token length.

DiffusionGemma-26B-A4B is intentionally NOT shown: it is block-AR (prompted
per-site naming, no in-place mask canvas), so the mask-token count is undefined.

Rebuilt with the unified CoReFusion blue/orange style.
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from corefusion_style import (BLUE, ORANGE, INK, GRAY, GRID,
    apply_style, savefig)
apply_style()

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DREAMON_SUMMARY = os.path.join(REPO, "results", "1t5t_exp", "dreamon_mask_ablation_summary.csv")

CYAN = "#00A6D6"   # TU Delft cyan — third dLLM (DreamOn)

# ── data ────────────────────────────────────────────────────────────────────
# Panel (a): EM (%) vs mask tokens per identifier k  (first-site EM)
K = [1, 2, 3, 4, 5]
EM_DIFFU = [26.2, 31.0, 27.2, 16.2, 13.6]   # DiffuCoder-7B (blue)   — published
EM_DREAM = [24.5, 34.8, 26.0, 18.8, 15.6]   # DreamCoder-7B (orange) — published


def load_dreamon_em():
    """DreamOn first-site EM per k from the ablation summary, or None if absent."""
    if not os.path.exists(DREAMON_SUMMARY):
        return None
    by_k = {}
    with open(DREAMON_SUMMARY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("model") == "DreamOn-7B":
                try:
                    by_k[int(r["k"])] = float(r["em"])
                except (KeyError, ValueError):
                    pass
    if not by_k:
        return None
    return [by_k.get(k, np.nan) for k in K]


EM_DREAMON = load_dreamon_em()
SERIES = [("DiffuCoder-7B", EM_DIFFU, BLUE), ("DreamCoder-7B", EM_DREAM, ORANGE)]
if EM_DREAMON is not None:
    SERIES.append(("DreamOn-7B", EM_DREAMON, CYAN))
    print("DreamOn EM(k):", EM_DREAMON)
else:
    print(f"[note] DreamOn ablation summary not found ({DREAMON_SUMMARY}); "
          f"plotting DiffuCoder/DreamCoder only.\n"
          f"       run experiments/1t5t_exp/part2_dreamon_mask_ablation.py on DAIC.")

# Panel (b): GT identifier length distribution (% of samples)
LEN_LABELS = ["1", "2", "3", "4", "5", "6+"]
LEN_PCT    = [38.1, 38.6, 15.7, 4.8, 2.0, 0.7]

# ── figure ──────────────────────────────────────────────────────────────────
fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                               gridspec_kw={"width_ratios": [1.25, 1.0]})

# ── Panel (a): grouped bars ─────────────────────────────────────────────────
x = np.arange(len(K))
n = len(SERIES)
w = 0.80 / n
offsets = (np.arange(n) - (n - 1) / 2) * w

for (name, vals, col), off in zip(SERIES, offsets):
    bars = axa.bar(x + off, vals, w, color=col, label=name,
                   edgecolor="white", linewidth=0.6, zorder=3)
    for rect, v in zip(bars, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        axa.annotate(f"{v:.1f}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     xytext=(0, 2.0), textcoords="offset points", ha="center",
                     va="bottom", fontsize=7.0, color=INK)

axa.set_xticks(x)
axa.set_xticklabels([str(k) for k in K])
axa.set_xlabel("Mask tokens per identifier $k$")
axa.set_ylabel("Exact Match (%)")
axa.set_ylim(0, 40)
axa.set_yticks(np.arange(0, 41, 10))
axa.set_xlim(-0.6, len(K) - 0.4)
axa.legend(loc="upper right", fontsize=8.5, borderaxespad=0.5, handlelength=1.3,
           ncol=1)
axa.set_title("(a) EM vs. mask tokens per identifier", fontsize=11, pad=8)
# DiffusionGemma caveat
axa.annotate("DiffusionGemma-26B: N/A (block-AR, no mask canvas)",
             xy=(0.015, 0.02), xycoords="axes fraction", ha="left", va="bottom",
             fontsize=6.8, style="italic", color=GRAY)

# ── Panel (b): GT length distribution ───────────────────────────────────────
xb = np.arange(len(LEN_LABELS))
bc = axb.bar(xb, LEN_PCT, 0.66, color=BLUE, edgecolor="white",
             linewidth=0.6, zorder=3)
for rect, v in zip(bc, LEN_PCT):
    axb.annotate(f"{v:.1f}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                 xytext=(0, 2.5), textcoords="offset points", ha="center",
                 va="bottom", fontsize=8, color=INK)

axb.set_xticks(xb)
axb.set_xticklabels(LEN_LABELS)
axb.set_xlabel("GT identifier length (sub-word tokens)")
axb.set_ylabel("% of samples")
axb.set_ylim(0, 45)
axb.set_yticks(np.arange(0, 46, 10))
axb.set_xlim(-0.6, len(LEN_LABELS) - 0.4)
axb.set_title("(b) GT identifier length distribution", fontsize=11, pad=8)

fig.tight_layout(w_pad=2.2)
savefig(fig, "fig06_mask_count_ablation")
plt.close(fig)
