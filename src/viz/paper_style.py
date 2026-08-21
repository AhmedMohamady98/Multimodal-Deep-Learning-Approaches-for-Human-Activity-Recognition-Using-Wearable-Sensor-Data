"""
Figure style for v3 thesis figures.

A hybrid look chosen after side-by-side review:
  - Editorial palette for fusion methods (jewel tones: deep teal, mustard,
    terracotta) — refined research-paper feel.
  - Activity palette = `tab20` — visual match to HARMES paper Fig 1 legend.
  - Modality colors pulled from HARMES paper Fig 5 architecture diagram.
  - Confusion matrices use `Blues` cmap (paper Fig 6 convention).
  - Helvetica/Arial font, no top/right spines, ticks on left/bottom only.
  - All figures saved as SVG + PDF (vector, lossless) for thesis inclusion.

Usage:

    from src.viz import apply_paper_style, FUSION_COLORS, CM_CMAP, save_fig
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    ...
    save_fig(fig, "results/v3/10s/fusion/overview/leaderboard")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import ACTIVITY_TO_ID, ID_TO_ACTIVITY
from src.data.paper_names import CODE_TO_PAPER, paper_label_for_id


# ── Palettes ────────────────────────────────────────────────────────────────

# Modality colors — matched to Fig 5 of the paper.
MODALITY_COLORS: dict[str, str] = {
    "IMU":      "#F4B860",   # warm yellow (paper IMU branch)
    "Audio":    "#5DAE8B",   # green-teal (paper audio branch)
    "Humidity": "#6FA8DC",   # soft blue (paper humidity branch)
}

# Fusion-method colors — editorial jewel-tone palette (chosen by user
# preference after style review). Each color is distinct enough to read
# across 7 bars; collectively the palette has a refined research-paper feel.
FUSION_COLORS: dict[str, str] = {
    "Late Fusion":     "#264653",   # deep teal
    "GMF":             "#2A9D8F",   # teal
    "MBT":             "#E9C46A",   # mustard
    "CMA":             "#F4A261",   # warm orange
    "LMF":             "#E76F51",   # terracotta
    "Decision Fusion": "#8E7AB5",   # soft purple
    "CLS Transformer": "#B5838D",   # dusty rose
}

# Confusion-matrix colormaps.
#   CM_CMAP      — for raw / normalized confusion matrices (every method, every fold).
#                  Blues matches the HARMES paper Fig 6 convention.
#   CM_DIFF_CMAP — for difference / delta matrices (e.g. 3-mod − 2-mod ablation).
#                  Divergent: RdBu_r puts HIGH values in red, LOW in blue, zero
#                  white. Whether red means better or worse depends entirely on
#                  what is being differenced, so state it per figure and check it
#                  against the data. For an F1 delta, red is the higher F1 and so
#                  the better one; for an off-diagonal confusion count, red is
#                  more confusion and so the worse one.
CM_CMAP: str = "Blues"
CM_DIFF_CMAP: str = "RdBu_r"

# Fold colors — three distinct jewel tones for 3-fold CV plots.
FOLD_COLORS: list[str] = ["#264653", "#E9C46A", "#E76F51"]   # teal / mustard / terracotta

# Activity palette — visual match to Fig 1 legend (tab20 has 20 entries,
# we take the first 16 in label order). Keys are paper-aligned activity names.
_tab20 = sns.color_palette("tab20", n_colors=20)
ACTIVITY_COLORS: dict[str, tuple[float, float, float]] = {
    paper_label_for_id(i): _tab20[i] for i in range(len(ID_TO_ACTIVITY))
}


# ── rcParams ────────────────────────────────────────────────────────────────

def apply_paper_style() -> None:
    """Set matplotlib rcParams to match the HARMES paper's figure aesthetic.

    Safe to call multiple times. Notebooks should call this once at the top.
    """
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    mpl.rcParams.update({
        # ── Fonts (Helvetica preferred, broad fallback chain) ───────────────
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial",
                                 "Liberation Sans", "DejaVu Sans"],
        "font.size":            9.5,
        "axes.titlesize":       10.5,
        "axes.titleweight":     "semibold",
        "axes.labelsize":       9.5,
        "axes.labelweight":     "normal",
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "legend.fontsize":      8.5,
        "legend.title_fontsize": 9,

        # ── Axes & spines ───────────────────────────────────────────────────
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       1.0,
        "axes.edgecolor":       "#1a1a1a",
        "axes.labelcolor":      "#222222",

        # ── Ticks ───────────────────────────────────────────────────────────
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     4.0,
        "ytick.major.size":     4.0,
        "xtick.major.width":    1.0,
        "ytick.major.width":    1.0,
        "xtick.color":          "#1a1a1a",
        "ytick.color":          "#1a1a1a",

        # ── Grid (subtle — paper uses no grid in most figs) ─────────────────
        "axes.grid":            False,
        "grid.color":           "#E0E0E0",
        "grid.linewidth":       0.5,

        # ── Legend ──────────────────────────────────────────────────────────
        "legend.frameon":       False,
        "legend.borderaxespad": 0.4,

        # ── Figure ──────────────────────────────────────────────────────────
        "figure.dpi":           120,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        "savefig.transparent":  False,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",

        # ── Lines & patches ─────────────────────────────────────────────────
        "lines.linewidth":      1.4,
        "patch.linewidth":      0.5,
        "patch.edgecolor":      "#333333",
    })


# ── Saving ──────────────────────────────────────────────────────────────────

def save_fig(
    fig: plt.Figure,
    path_no_ext: str | os.PathLike,
    *,
    formats: tuple[str, ...] = ("svg", "pdf"),
    close: bool = True,
) -> list[Path]:
    """Save a matplotlib Figure to multiple vector formats.

    Parameters
    ----------
    fig
        The Figure object.
    path_no_ext
        Output path WITHOUT extension — e.g. `results/v3/10s/fusion/overview/leaderboard`.
        Parent directory is created automatically.
    formats
        File formats to write. Default = SVG + PDF (both vector, no quality loss).
        Add "png" if you also want a quick-preview raster.
    close
        Close the figure after saving (default True — avoids matplotlib memory bloat
        in notebooks that produce many figures).

    Returns
    -------
    List of written file paths.
    """
    base = Path(path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        written.append(out)
    if close:
        plt.close(fig)
    return written


# ── Convenience helpers ─────────────────────────────────────────────────────

def activity_color(name_or_id) -> tuple[float, float, float]:
    """Look up the activity color by paper name, codebase name, or integer ID."""
    if isinstance(name_or_id, int):
        return ACTIVITY_COLORS[paper_label_for_id(name_or_id)]
    if name_or_id in ACTIVITY_COLORS:
        return ACTIVITY_COLORS[name_or_id]
    if name_or_id in CODE_TO_PAPER:
        return ACTIVITY_COLORS[CODE_TO_PAPER[name_or_id]]
    raise KeyError(f"Unknown activity: {name_or_id!r}")


if __name__ == "__main__":
    # Smoke test: write a tiny palette swatch PDF/SVG so we can eyeball the look.
    import numpy as np
    apply_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))

    # Modality swatches
    axes[0].bar(MODALITY_COLORS.keys(), [1, 1, 1], color=MODALITY_COLORS.values())
    axes[0].set_title("Modalities")
    axes[0].set_yticks([])

    # Fusion swatches
    axes[1].bar(range(len(FUSION_COLORS)), [1] * len(FUSION_COLORS),
                color=list(FUSION_COLORS.values()))
    axes[1].set_xticks(range(len(FUSION_COLORS)))
    axes[1].set_xticklabels(list(FUSION_COLORS.keys()), rotation=30, ha="right")
    axes[1].set_title("Fusion methods")
    axes[1].set_yticks([])

    # Activity swatches (16 thin bars)
    names = list(ACTIVITY_COLORS.keys())
    axes[2].bar(range(len(names)), [1] * len(names),
                color=list(ACTIVITY_COLORS.values()))
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    axes[2].set_title("Activities (paper Fig 1 palette)")
    axes[2].set_yticks([])

    fig.suptitle("HARMES paper-aligned style — palette preview", y=1.04)
    save_fig(fig, "results/v3/_style_preview", formats=("svg", "pdf", "png"))
    print("Wrote results/v3/_style_preview.{svg,pdf,png}")
