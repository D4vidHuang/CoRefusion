"""
CoReFusion unified figure style — TU Delft blue + orange.

Every rebuilt thesis figure imports from here so the whole document shares one
palette and one set of typographic / axis defaults.

Locked palette (from figures/new/MAPPING.md):
    BLUE   = #0076C2   (TU Delft)   ->  DiffuCoder / AR / Seq2Seq / "clean"
    ORANGE = #FF8000                ->  DreamCoder / dLLM / "smelly"

Usage
-----
    from corefusion_style import (
        BLUE, ORANGE, INK, GRAY, GRID,
        apply_style, blue_ramp, orange_ramp, savefig,
        TWO_MODEL, ARCH, SMELL,
    )
    apply_style()
    ...
    savefig(fig, "fig05_em_vs_cardinality")
"""
from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# ---------------------------------------------------------------- core palette
BLUE   = "#0076C2"   # TU Delft primary blue
ORANGE = "#FF8000"   # accent orange

# darker / lighter anchors of each family (for ramps, fills, emphasis)
BLUE_DARK    = "#00396B"
BLUE_MID     = "#0064A8"
BLUE_LIGHT   = "#5FAEDD"
BLUE_PALE    = "#BFE0F2"

ORANGE_DARK  = "#B35900"
ORANGE_MID   = "#E67300"
ORANGE_LIGHT = "#FFA94D"
ORANGE_PALE  = "#FFE0C2"

# neutrals
INK        = "#222222"   # primary text / strong lines
GRAY       = "#6E6E6E"   # secondary text / neutral series
LIGHT_GRAY = "#B7B7B7"
GRID       = "#E4E8EC"   # gridlines
PANEL      = "#F4F6F8"   # subtle panel fill
WHITE      = "#FFFFFF"

# semantic aliases used across figures -------------------------------------
TWO_MODEL = {"DiffuCoder-7B": BLUE, "DreamCoder-7B": ORANGE,
             "DiffuCoder": BLUE, "DreamCoder": ORANGE}
ARCH      = {"ar": BLUE, "seq2seq": BLUE, "dllm": ORANGE,
            "AR": BLUE, "Seq2Seq": BLUE, "dLLM": ORANGE}
SMELL     = {"clean": BLUE, "smelly": ORANGE, "context": BLUE, "smell": ORANGE,
             "gt": BLUE, "ground_truth": BLUE}

# colormaps (continuous) ----------------------------------------------------
CMAP_BLUE   = LinearSegmentedColormap.from_list("cf_blue",   ["#E7F2FA", BLUE,   BLUE_DARK])
CMAP_ORANGE = LinearSegmentedColormap.from_list("cf_orange", ["#FFF1E3", ORANGE, ORANGE_DARK])
# diverging blue<->orange, white centre (e.g. heatmaps, similarity)
CMAP_DIV    = LinearSegmentedColormap.from_list(
    "cf_div", [BLUE_DARK, BLUE, "#F4F6F8", ORANGE, ORANGE_DARK])


def blue_ramp(n: int):
    """n perceptually-spaced blues, dark -> light (good for ordered series)."""
    if n <= 1:
        return [BLUE]
    return [CMAP_BLUE(0.30 + 0.62 * i / (n - 1)) for i in range(n)]


def orange_ramp(n: int):
    if n <= 1:
        return [ORANGE]
    return [CMAP_ORANGE(0.32 + 0.60 * i / (n - 1)) for i in range(n)]


def lighten(color, amount: float):
    """Blend `color` toward white by `amount` in [0,1]."""
    r, g, b, a = to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, a)


# ------------------------------------------------------------------ rcParams
def apply_style(base: float = 10.0):
    plt.rcParams.update({
        # typography — sans, matches the thesis body
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                             "TeX Gyre Heros", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        # color of text / axes
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#9AA0A6",
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.titlecolor": INK,
        # spines: keep left+bottom only
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        # grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        # lines / markers
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 0.0,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.4,
        # figure / saving
        "figure.dpi": 120,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.dpi": 300,
        "savefig.facecolor": WHITE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,   # editable text in the PDF
        "ps.fonttype": 42,
    })


# default output location
OUTDIR = os.path.dirname(os.path.abspath(__file__))


def savefig(fig, stem: str, outdir: str | None = None, png: bool = True, pdf: bool = True):
    """Save `fig` as <stem>.pdf and <stem>.png into the rebuilt-figures dir."""
    outdir = outdir or OUTDIR
    os.makedirs(outdir, exist_ok=True)
    paths = []
    if pdf:
        p = os.path.join(outdir, stem + ".pdf"); fig.savefig(p); paths.append(p)
    if png:
        p = os.path.join(outdir, stem + ".png"); fig.savefig(p, dpi=300); paths.append(p)
    print("saved:", *paths, sep="\n  ")
    return paths


if __name__ == "__main__":
    # tiny swatch sheet to eyeball the palette
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 2.2))
    swatches = [("BLUE", BLUE), ("BLUE_MID", BLUE_MID), ("BLUE_DARK", BLUE_DARK),
                ("ORANGE", ORANGE), ("ORANGE_MID", ORANGE_MID), ("ORANGE_DARK", ORANGE_DARK),
                ("INK", INK), ("GRAY", GRAY), ("GRID", GRID)]
    for i, (name, c) in enumerate(swatches):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=c))
        ax.text(i + 0.46, -0.18, name, ha="center", va="top", fontsize=7, rotation=30)
    ax.set_xlim(-0.2, len(swatches)); ax.set_ylim(-0.7, 1.1); ax.axis("off")
    ax.set_title("CoReFusion palette")
    savefig(fig, "_palette_swatches")
