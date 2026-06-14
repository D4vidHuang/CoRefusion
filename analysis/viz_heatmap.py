import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE = "#0076C2", "#FF8000"
OUT = "/Users/davidhuang/Desktop/CoRefusion/figures/new"
CSV = f"{OUT}/leaderboard_full.csv"

# display columns: (csv_key, header). all "higher = better".
COLS = [("em_gated","EM"), ("em_consistent","EM_c"), ("consistency","cons"),
        ("lev","lev"), ("nw","nw"), ("fuzzy","fuzzy"), ("qual","qual"),
        ("LJ_Q7","Q7"), ("LJ_Q14","Q14"), ("LJ_Q32","Q32"),
        ("LJ_M24","M24"), ("LJ_G27","G27")]

rows = []
for r in csv.DictReader(open(CSV)):
    if r["model"] == "DiffusionGemma-26B-A4B":
        continue
    f = lambda k: float(r[k]) * 100 if r.get(k) not in (None, "", "None") else 0.0
    rows.append({"model": r["model"], "is_dllm": "dLLM" in r["arch"],
                 **{k: f(k) for k, _ in COLS}})
rows.sort(key=lambda r: -r["em_gated"])               # best on top

models = [r["model"] for r in rows]
M = np.array([[r[k] for k, _ in COLS] for r in rows])  # raw values (%)
# per-column min-max normalization (best=1, worst=0)
mn, mx = M.min(0), M.max(0)
N = (M - mn) / np.where(mx - mn == 0, 1, mx - mn)

cmap = LinearSegmentedColormap.from_list("bwo", [BLUE, "#f7f7f7", ORANGE])

fig, ax = plt.subplots(figsize=(11.5, 9.5))
im = ax.imshow(N, cmap=cmap, aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([h for _, h in COLS], fontsize=9.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
for tick, r in zip(ax.get_yticklabels(), rows):       # color model name by family
    tick.set_color(ORANGE if r["is_dllm"] else "#222")
    if r["is_dllm"]: tick.set_fontweight("bold")
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

# annotate raw values; text color by cell luminance
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        rr, gg, bb, _ = cmap(N[i, j])
        lum = 0.299*rr + 0.587*gg + 0.114*bb
        ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                fontsize=6.8, color="white" if lum < 0.5 else "#222")

# section dividers between metric groups
for x in (2.5, 6.5):                                  # after EM-group, after M-group
    ax.axvline(x, color="white", lw=2)

cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("per-column rank  (blue = worst · orange = best)", fontsize=9)
cb.set_ticks([0, 1]); cb.set_ticklabels(["worst", "best"])

fig.text(0.5, 0.005,
         "EM=strict all-sites · EM_c=EM on consistent subset · cons=consistency rate · "
         "lev/nw=M1 · fuzzy=M2 · qual=M3 · Q7/Q14/Q32=Qwen2.5 · M24=Mistral-Small-24B · G27=Gemma-2-27B",
         ha="center", fontsize=7.5, color="#777")
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(f"{OUT}/fig7_metric_heatmap.png", bbox_inches="tight", dpi=150)
print("wrote", f"{OUT}/fig7_metric_heatmap.png")
