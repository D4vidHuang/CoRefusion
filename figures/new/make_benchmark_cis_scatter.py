"""
Benchmark scatter, regenerated with the NEW fused metric (CIS) alongside the
OLD Exact-Match (EM) and LLM-judge (LJ).

All three metrics are computed on the SAME predictions (the unified_refineID
judge CSVs, strict all-sites / gated denominator), so they are directly
comparable:
  EM  = strict all-sites exact match (gated; inconsistent/non-fill -> 0)
        [reproduces lj_vs_em_table.csv]
  LJ  = LLM-judge acceptance (Qwen2.5-7B-Instruct = thesis default; gated)
  CIS = CoReFusion Identifier Score = mean(lev,nw,jaccard,fuzzy), gated.

Outputs (figures/new/):
  benchmark_cis_scatter.png/pdf   2-panel  LJ-vs-EM | LJ-vs-CIS
  benchmark_metric_ranks.png/pdf  EM -> CIS -> LJ ranking bump chart
  leaderboard_benchmark_cis.csv   the numbers
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

BLUE, ORANGE, MUTED = "#0076C2", "#FF8000", "#5C6B78"
INK, LINE, PANEL = "#14202B", "#D9E2EA", "#F2F6FA"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": LINE,
                     "font.size": 13, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5})

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = ["lev_sim", "nw_sim", "subtok_jaccard", "subtok_fuzzy"]
PARAMS = {"DreamCoder-7B": 7, "DiffuCoder-7B": 7, "DreamOn-7B": 7, "CodeLlama-13B": 13,
          "CodeLlama-7B": 7, "StarCoder2-15B": 15, "StarCoder2-7B": 7, "StarCoder2-3B": 3,
          "Qwen2.5-Coder-14B": 14, "Qwen2.5-Coder-7B": 7, "Qwen2.5-Coder-3B": 3,
          "Qwen2.5-Coder-1.5B": 1.5, "DeepSeek-Coder-6.7B": 6.7, "DeepSeek-Coder-1.3B": 1.3,
          "CodeGemma-7B": 7, "CodeGemma-2B": 2, "CodeT5p-16B": 16, "CodeT5p-6B": 6,
          "CodeT5p-2B": 2, "CodeT5-large": 0.77, "CodeT5-base": 0.22, "CodeT5-small": 0.06}
FAM_STY = {"dLLM": (BLUE, "o"), "FIM": (MUTED, "s"), "Seq2Seq": (ORANGE, "^")}
FAM_NAME = {"dLLM": "Diffusion LLM (dLLM)", "FIM": "Decoder-only (FIM)", "Seq2Seq": "Encoder-decoder"}

# ---- build per-model EM / LJ / CIS on the same predictions ----
lg = pd.read_csv(os.path.join(ROOT, "results/identifier_metrics/fusion_long.csv"))
lg["CIS"] = lg[SIM].mean(axis=1)  # already 0 for inconsistent rows
q7 = lg[lg.judge == "Qwen2.5-7B-Instruct"]
lb = q7.groupby(["model", "family"]).agg(EM=("em", "mean"), LJ=("lj", "mean"),
                                         CIS=("CIS", "mean"), n=("em", "size")).reset_index()
# consensus LJ (majority of up to 5 judges) as a robustness column
cons = pd.read_csv(os.path.join(ROOT, "results/identifier_metrics/fusion_consensus.csv"))
cons["ljm"] = np.where(cons.lj_n_judges > 0, (cons.lj_majority > 0).astype(float), np.nan)
ljc = cons.groupby("model").ljm.mean().rename("LJ_consensus")
lb = lb.merge(ljc, on="model")
for c in ["EM", "LJ", "CIS", "LJ_consensus"]:
    lb[c] *= 100
lb["params"] = lb.model.map(PARAMS)
lb = lb[lb.family.isin(FAM_STY)].copy()   # drop DiffusionGemma-26B (all-zero, not in the 22-model benchmark)
lb = lb.sort_values("LJ", ascending=False).reset_index(drop=True)
lb.to_csv(os.path.join(HERE, "leaderboard_benchmark_cis.csv"), index=False)

rho_em = stats.spearmanr(lb.EM, lb.LJ).correlation
rho_cis = stats.spearmanr(lb.CIS, lb.LJ).correlation
r_em = stats.pearsonr(lb.EM, lb.LJ)[0]
r_cis = stats.pearsonr(lb.CIS, lb.LJ)[0]
print(lb.round(1).to_string(index=False))
print(f"\nSpearman(.,LJ): EM={rho_em:.3f} CIS={rho_cis:.3f} | Pearson: EM={r_em:.3f} CIS={r_cis:.3f}")

sz = lambda p: 60 + 90 * np.sqrt(p)

def panel(ax, xcol, xlabel, rho, r):
    lo = 0
    hi = max(lb[xcol].max(), lb.LJ.max()) * 1.12
    ax.fill_between([lo, hi], [lo, hi], hi, color=PANEL, zorder=0)        # LJ > metric region
    ax.plot([lo, hi], [lo, hi], ls="--", color=MUTED, lw=1.2, zorder=1)
    ax.text(hi * 0.82, hi * 0.9, "LJ = x", color=MUTED, fontsize=12, rotation=45, ha="center", va="center")
    for fam, (col, mk) in FAM_STY.items():
        s = lb[lb.family == fam]
        ax.scatter(s[xcol], s.LJ, s=[sz(p) for p in s.params], c=col, marker=mk,
                   alpha=0.82, edgecolors="white", linewidths=1.1, zorder=4, label=FAM_NAME[fam])
    # fitted trend (LJ ~ x)
    b, a = np.polyfit(lb[xcol], lb.LJ, 1)
    xs = np.array([lo, hi]); ax.plot(xs, a + b * xs, color=INK, lw=1.4, ls="-", alpha=0.65, zorder=3)
    # label the dLLMs (the interesting ones). DreamCoder & DiffuCoder sit almost
    # on top of each other near the top -> stagger their labels so they don't collide.
    dllm_off = {"DreamCoder-7B": (0.02, 0.065), "DiffuCoder-7B": (0.02, -0.085),
                "DreamOn-7B": (0.025, 0.035)}
    for _, r0 in lb[lb.family == "dLLM"].iterrows():
        dx, dy = dllm_off.get(r0.model, (0.02, 0.03))
        ax.annotate(r0.model, (r0[xcol], r0.LJ),
                    xytext=(r0[xcol] + hi * dx, r0.LJ + hi * dy),
                    fontsize=11, color=INK, zorder=6, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=3))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=14.5); ax.set_ylabel("LLM-judge  LJ (%)", fontsize=14.5)
    ax.grid(True, color=LINE, lw=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\nPearson r = {r:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, color=INK,
            bbox=dict(boxstyle="round", fc="white", ec=LINE))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.4))
panel(axL, "EM", "Exact Match  EM (%)", rho_em, r_em)
panel(axR, "CIS", "CoReFusion Identifier Score  CIS (%)", rho_cis, r_cis)
axL.set_title("OLD: LJ vs Exact Match", fontsize=15, fontweight="bold")
axR.set_title("NEW: LJ vs fused CIS", fontsize=15, fontweight="bold")
axL.legend(loc="lower right", frameon=True, fontsize=11.5, facecolor="white", edgecolor=LINE,
           title="Architecture", title_fontsize=12.5)
fig.suptitle("Benchmark landscape (RQ1): the fused CIS is a far tighter proxy for the expensive LLM-judge than EM",
             fontsize=15.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.02, "marker size ∝ parameters · grey band = region where LJ accepts more than the metric credits · "
         "all three metrics from the same predictions (strict all-sites, gated)",
         ha="center", fontsize=10.5, color=MUTED, style="italic")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "benchmark_cis_scatter.png"), dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(os.path.join(HERE, "benchmark_cis_scatter.pdf"), facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---- ranking bump chart: EM -> CIS -> LJ ----
order_lj = lb.sort_values("LJ", ascending=False).model.tolist()
# method="first" breaks ties into UNIQUE ranks 1..N so two tied models never
# land on the same y (which is what made the row labels overlap).
rk = pd.DataFrame({
    "EM": lb.set_index("model").EM.rank(ascending=False, method="first"),
    "CIS": lb.set_index("model").CIS.rank(ascending=False, method="first"),
    "LJ": lb.set_index("model").LJ.rank(ascending=False, method="first")})
cols3 = ["EM", "CIS", "LJ"]
fig2, ax = plt.subplots(figsize=(8.6, 9))
for m in order_lj:
    fam = lb.set_index("model").family[m]
    col = FAM_STY[fam][0]
    ys = [rk.loc[m, c] for c in cols3]
    ax.plot(range(3), ys, "-o", color=col, alpha=0.85, lw=1.8, ms=6, zorder=3)
    ax.text(-0.06, rk.loc[m, "EM"], m, ha="right", va="center", fontsize=8.2, color=INK)
    ax.text(2.06, rk.loc[m, "LJ"], m, ha="left", va="center", fontsize=8.2, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(["EM rank", "CIS rank", "LJ rank"], fontsize=11)
ax.set_ylim(len(lb) + 0.5, 0.5); ax.set_yticks([])
ax.set_ylabel("rank (1 = best, top)")
ax.set_xlim(-0.75, 2.75)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.set_title("Model ranking under EM, CIS, and LJ\nCIS rank ≈ LJ rank (ρ=%.2f); EM rank diverges (ρ=%.2f)" % (rho_cis, rho_em),
             fontsize=12.5, fontweight="bold")
ax.legend(handles=[Patch(fc=c, label=FAM_NAME[f]) for f, (c, _) in FAM_STY.items()],
          loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(HERE, "benchmark_metric_ranks.png"), dpi=200, facecolor="white", bbox_inches="tight")
fig2.savefig(os.path.join(HERE, "benchmark_metric_ranks.pdf"), facecolor="white", bbox_inches="tight")
plt.close(fig2)
print("\nwrote figures/new/benchmark_cis_scatter.{png,pdf}, benchmark_metric_ranks.{png,pdf}, leaderboard_benchmark_cis.csv")
