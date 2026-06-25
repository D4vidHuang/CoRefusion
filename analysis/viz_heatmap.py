import csv
import os
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, GREEN = "#0076C2", "#FF8000", "#118A2E"
OUT = "/Users/davidhuang/Desktop/CoRefusion/figures/new"
CSV = f"{OUT}/leaderboard_full.csv"
REPO = "/Users/davidhuang/Desktop/CoRefusion"
CEILING_CSV = f"{REPO}/results/refineid_groundtruth_judge/ceiling_summary.csv"
TEST_CSV = f"{REPO}/data/test.csv"

# display columns: (csv_key, header). all "higher = better".
COLS = [("em_gated","EM"), ("em_consistent","EM_c"), ("consistency","cons"),
        ("lev","lev"), ("nw","nw"), ("fuzzy","fuzzy"), ("qual","qual"),
        ("LJ_Q7","Q7"), ("LJ_Q14","Q14"), ("LJ_Q32","Q32"),
        ("LJ_M24","M24"), ("LJ_G27","G27")]

# judge name (in ceiling_summary.csv) -> LJ column key in this heatmap
JUDGE2COL = {"Qwen2.5-7B-Instruct": "LJ_Q7", "Qwen2.5-14B-Instruct": "LJ_Q14",
             "Qwen2.5-32B-Instruct": "LJ_Q32", "Mistral-Small-24B": "LJ_M24",
             "Gemma-2-27B-It": "LJ_G27"}

# ---------------------------------------------------------------------------
# benchmarked models (one row each)
# ---------------------------------------------------------------------------
def _family(arch):
    a = (arch or "").lower()
    if "dllm" in a:
        return "dLLM"
    if "encoder" in a or "seq2seq" in a or "t5" in a:
        return "Seq2Seq"
    return "AR"

# row order: group by architecture (dLLM -> AR -> Seq2Seq), alphabetical within group
FAM_ORDER = {"dLLM": 0, "AR": 1, "Seq2Seq": 2}

rows = []
for r in csv.DictReader(open(CSV)):
    f = lambda k: float(r[k]) * 100 if r.get(k) not in (None, "", "None") else 0.0
    vals = {k: f(k) for k, _ in COLS}
    # Skip all-zero rows (e.g. a model whose predictions are all empty): a row
    # of zeros has no signal and would distort the per-column min-max scaling.
    # DiffusionGemma auto-appears once its predictions are non-empty.
    if max(vals.values()) == 0:
        continue
    rows.append({"model": r["model"], "family": _family(r["arch"]), **vals})
rows.sort(key=lambda r: (FAM_ORDER[r["family"]], r["model"].lower()))

models = [r["model"] for r in rows]
M = np.array([[r[k] for k, _ in COLS] for r in rows])  # raw values (%)
# per-column min-max normalization computed over BENCHMARKED MODELS ONLY, so the
# ground-truth ceiling row added below does NOT rescale the existing model cells.
mn, mx = M.min(0), M.max(0)
N = (M - mn) / np.where(mx - mn == 0, 1, mx - mn)

# ---------------------------------------------------------------------------
# Ground-truth ceiling row: judge acceptance of the dataset's own gold names
# (from experiments/judge_refineid_ground_truth.py -> analyze_gt_judge_ceiling.py).
# EM / similarity columns are 1.0 by construction (a perfect prediction IS the
# gold name); qual = the gold names' intrinsic readability (M3, computed below).
# ---------------------------------------------------------------------------
def load_ceiling_row():
    if not os.path.exists(CEILING_CSV):
        sys.exit(f"missing {CEILING_CSV}\n  run: python analysis/analyze_gt_judge_ceiling.py")
    lj = {}
    for r in csv.DictReader(open(CEILING_CSV)):
        col = JUDGE2COL.get(r["judge"])
        if col and r.get("gt_accept_rate") not in (None, "", "None"):
            lj[col] = float(r["gt_accept_rate"]) * 100
    missing = [c for c in JUDGE2COL.values() if c not in lj]
    if missing:
        print(f"  [warn] ceiling missing judges for cols {missing} -> shown as n/a")
    # intrinsic quality of the gold identifiers (same M3 as the leaderboard 'qual')
    sys.path.insert(0, os.path.join(REPO, "analysis"))
    import identifier_similarity_metrics as IM
    d = IM.load_dictionary()
    csv.field_size_limit(2**31 - 1)
    gts = [row[2] for row in csv.reader(open(TEST_CSV))]
    gt_qual = sum(IM.metric_quality(g, d)["qual_char"] for g in gts) / len(gts) * 100

    vals = {}
    for k, _ in COLS:
        if k in ("em_gated", "em_consistent", "consistency", "lev", "nw", "fuzzy"):
            vals[k] = 100.0                 # a perfect prediction equals the gold name
        elif k == "qual":
            vals[k] = gt_qual               # intrinsic readability of the gold names
        else:                               # LJ_* columns
            vals[k] = lj.get(k, np.nan)
    return vals

gt_vals = load_ceiling_row()
gt_vec = np.array([gt_vals[k] for k, _ in COLS])
# colour the ceiling row on the SAME model scale, clipped to [0,1] (NaN -> shown grey)
with np.errstate(invalid="ignore"):
    N_gt = np.clip((gt_vec - mn) / np.where(mx - mn == 0, 1, mx - mn), 0, 1)

# stack ceiling row ON TOP of the models
M_disp = np.vstack([gt_vec, M])
N_disp = np.vstack([N_gt, N])
labels = ["★ Ground Truth"] + models
fam_list = [None] + [r["family"] for r in rows]        # None marks the ceiling row

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list("bwo", [BLUE, "#f7f7f7", ORANGE])

fig, ax = plt.subplots(figsize=(11.5, 9.9))
im = ax.imshow(N_disp, cmap=cmap, aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([h for _, h in COLS], fontsize=9.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
FAM_COLOR = {"dLLM": ORANGE, "AR": "#222", "Seq2Seq": BLUE}
for tick, fam in zip(ax.get_yticklabels(), fam_list):  # colour model name by family
    if fam is None:                                    # the ground-truth ceiling row
        tick.set_color(GREEN); tick.set_fontweight("bold")
    else:
        tick.set_color(FAM_COLOR[fam])
        if fam == "dLLM":
            tick.set_fontweight("bold")
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

# annotate raw values; text colour by cell luminance (NaN cells -> 'n/a', grey bg)
for i in range(M_disp.shape[0]):
    for j in range(M_disp.shape[1]):
        v = M_disp[i, j]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#dddddd"))
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color="#666")
            continue
        rr, gg, bb, _ = cmap(N_disp[i, j])
        lum = 0.299*rr + 0.587*gg + 0.114*bb
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                fontsize=6.8, color="white" if lum < 0.5 else "#222")

# divider separating the ceiling row from the benchmarked models
ax.axhline(0.5, color="k", lw=2.5)
# dividers between architecture groups (dLLM | AR | Seq2Seq). model j is display row j+1.
_mfams = [r["family"] for r in rows]
for j in range(1, len(_mfams)):
    if _mfams[j] != _mfams[j - 1]:
        ax.axhline(j + 0.5, color="k", lw=1.4, alpha=0.55)
# section dividers between metric groups
for x in (2.5, 6.5):                                  # after EM-group, after M-group
    ax.axvline(x, color="white", lw=2)

cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("per-column rank  (blue = worst · orange = best)", fontsize=9)
cb.set_ticks([0, 1]); cb.set_ticklabels(["worst\n(0)", "best\n(1)"])

fig.tight_layout()
fig.savefig(f"{OUT}/fig7_metric_heatmap.png", bbox_inches="tight", dpi=150)
print("wrote", f"{OUT}/fig7_metric_heatmap.png")
