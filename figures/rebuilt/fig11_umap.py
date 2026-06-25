"""
Fig11 — UMAP projections of smell-position hidden states.

The raw hidden-state vectors are NOT saved (would need DiffuCoder-7B on GPU),
and experiment_a_internal_rep.py does NOT persist the UMAP 2D coordinates
(features_umap is computed in-memory and only the rendered PNG/PDF are written;
edit_signals_*.csv stores only cosine/L2 distances, not coords). So we take the
RECOLOR path: load the original rendered PNGs and hue-remap the matplotlib
default colours to the CoReFusion palette:

    smelly  red   ~#e74c3c (hue ~0)    ->  ORANGE  #FF8000
    gt/clean blue ~#3498db (hue ~204)  ->  BLUE    #0076C2

White/gray/black/text pixels are left untouched (hue masks keyed on saturation
and value). Anti-aliased edges are handled by blending toward the target hue
proportional to colourfulness so soft marker edges recolour smoothly.

Then the recoloured top (layer-wise) and bottom (diffusion-steps) panels are
vertically stacked into one figure.
"""
import sys
sys.path.insert(0, "/Users/davidhuang/Desktop/CoRefusion/figures/rebuilt")
from corefusion_style import BLUE, ORANGE, apply_style, savefig

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

apply_style()

SRC = "/Users/davidhuang/Desktop/CoRefusion/results/results_20260405_2056"
TOP = f"{SRC}/umap_layer_wise.png"      # top row  (across transformer layers)
BOT = f"{SRC}/umap_diffusion_steps.png"  # bottom row (across denoising steps)

# original matplotlib defaults present in the rendered PNGs
RED_HUE  = 0.0     # ~#e74c3c smelly
BLUE_HUE = 204.0   # ~#3498db gt / clean

TARGET_ORANGE = np.array(to_rgb(ORANGE))  # #FF8000
TARGET_BLUE   = np.array(to_rgb(BLUE))    # #0076C2


def rgb_to_hsv_arr(rgb):
    """Vectorised RGB->HSV for an (...,3) float array in [0,1]. H in degrees."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    df = mx - mn
    h = np.zeros_like(mx)
    # avoid div-by-zero
    nz = df > 1e-8
    # red is max
    mask = nz & (mx == r)
    h[mask] = (60 * ((g[mask] - b[mask]) / df[mask]) + 360) % 360
    # green is max
    mask = nz & (mx == g)
    h[mask] = (60 * ((b[mask] - r[mask]) / df[mask]) + 120) % 360
    # blue is max
    mask = nz & (mx == b)
    h[mask] = (60 * ((r[mask] - g[mask]) / df[mask]) + 240) % 360
    s = np.where(mx > 1e-8, df / np.where(mx > 1e-8, mx, 1.0), 0.0)
    v = mx
    return h, s, v


def hue_dist(h, center):
    """Circular hue distance in degrees."""
    d = np.abs(h - center)
    return np.minimum(d, 360 - d)


def recolor(path):
    """Remap red->ORANGE and blue->BLUE, leaving neutral pixels untouched."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    h, s, v = rgb_to_hsv_arr(im)

    out = im.copy()

    # Colourfulness weight: how strongly to pull a pixel to the target colour.
    # Fully saturated coloured cores -> ~1; near-white/gray edges -> small;
    # this smoothly recolours the anti-aliased marker fringe.
    # blend = s, clamped so genuine fills go all the way to target.
    blend = np.clip((s - 0.12) / (0.55 - 0.12), 0.0, 1.0)

    # --- red (smelly) -> ORANGE ---
    # accept a wide-ish wedge around hue 0 (wraps 350-360 too)
    red_mask = (hue_dist(h, RED_HUE) < 35) & (s > 0.12) & (v > 0.18)
    bw = blend[red_mask][..., None]
    out[red_mask] = (1 - bw) * im[red_mask] + bw * TARGET_ORANGE

    # --- blue (gt/clean) -> BLUE ---
    blue_mask = (hue_dist(h, BLUE_HUE) < 45) & (s > 0.12) & (v > 0.18)
    bw = blend[blue_mask][..., None]
    out[blue_mask] = (1 - bw) * im[blue_mask] + bw * TARGET_BLUE

    return (np.clip(out, 0, 1) * 255).astype(np.uint8), red_mask.sum(), blue_mask.sum()


top_img, n_red_t, n_blue_t = recolor(TOP)
bot_img, n_red_b, n_blue_b = recolor(BOT)
print(f"top  : recoloured {n_red_t:>8} red(->orange), {n_blue_t:>8} blue(->blue)")
print(f"bot  : recoloured {n_red_b:>8} red(->orange), {n_blue_b:>8} blue(->blue)")

# ---- vertically stack the two recoloured panels into one figure ----
# each source PNG is 3300x2100 px (11x7 in @ 300 dpi)
H, W = top_img.shape[0], top_img.shape[1]
fig_w = W / 300.0           # 11 in
fig_h = 2 * H / 300.0       # 14 in (two stacked panels)

fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))
axes[0].imshow(top_img)
axes[1].imshow(bot_img)
for ax in axes:
    ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.0)

savefig(fig, "fig11_umap")
plt.close(fig)
print("done")
