"""Shared helpers for every v3 notebook.

Import everything from here so notebook cells stay focused on the figures:

    from notebooks.v3._common import *

Loads JSONs from `results/v3/10s/`, applies the paper-aligned style,
exports activity labels in paper wording.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure `src` is importable regardless of where the notebook is run from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.viz import (  # noqa: E402
    apply_paper_style,
    FUSION_COLORS,
    MODALITY_COLORS,
    ACTIVITY_COLORS,
    FOLD_COLORS,
    CM_CMAP,
    CM_DIFF_CMAP,
    save_fig,
)
from src.data.dataset import ID_TO_ACTIVITY, NUM_CLASSES  # noqa: E402
from src.data.paper_names import PAPER_LABELS, paper_label_for_id  # noqa: E402


apply_paper_style()


# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = _REPO_ROOT / "results" / "v3" / "10s"
FIGURES_ROOT = RESULTS_DIR    # figures go inside results/v3/10s/<topic>/
TABLES_ROOT  = RESULTS_DIR    # tables alongside figures in the same topic folder


# ── Fusion method registry — display name → JSON file ───────────────────────
FUSION_METHODS: dict[str, str] = {
    "Late Fusion":     "late_fusion",
    "GMF":             "gmf",
    "MBT":             "mbt",
    "CMA":             "cma",
    "LMF":             "lmf",
    "Decision Fusion": "decision_fusion",
    "CLS Transformer": "cls_transformer",
}

UNIMODAL_MODELS = ["TinyHAR", "AST", "TSMixer"]
UNIMODAL_MODALITY = {"TinyHAR": "IMU", "AST": "Audio", "TSMixer": "Humidity"}

# Display name → key as it appears in the prediction CSV `model` column.
# Used by every notebook that loads `predictions/all_models_raw_predictions.csv`.
MODEL_CSV_KEYS: dict[str, str] = {
    "TinyHAR":         "tinyhar",
    "AST":             "ast",
    "TSMixer":         "tsmixer",
    "Late Fusion":     "late_fusion",
    "GMF":             "gmf",
    "MBT":             "mbt",
    "CMA":             "cma",
    "LMF":             "lmf",
    "Decision Fusion": "decision_fusion",
    "CLS Transformer": "cls_transformer",
}
ALL_MODELS_ORDER = list(MODEL_CSV_KEYS.keys())   # paper-ordered: unimodals first, then fusion

# Left-handed participants (HARMES paper Table 2). Used by left-hander
# and best-vs-worst notebooks for highlighting.
LEFT_HANDERS: set[int] = {7, 10, 14}


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_fusion_results(method_key: str) -> list[dict]:
    """Load a single fusion-method JSON (list of 3 fold dicts).

    `method_key` is the JSON filename stem, e.g. "gmf" or "late_fusion".
    Returns the parsed list. Raises FileNotFoundError with a helpful message
    if the file isn't there yet.
    """
    path = RESULTS_DIR / f"{method_key}_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            f"Run the cluster eval and pull v3 JSONs (rsync omni:multimodal-har-thesis/results/v3/10s/*.json results/v3/10s/)."
        )
    return json.loads(path.read_text())


def load_unimodal_results() -> dict[str, list[dict]]:
    """Load the unimodal-baseline JSON: {model_name: [fold0, fold1, fold2]}."""
    return json.loads((RESULTS_DIR / "unimodal_results.json").read_text())


def load_all_fusion() -> dict[str, list[dict]]:
    """Load every available fusion result. Quietly skips missing files."""
    out: dict[str, list[dict]] = {}
    for display, key in FUSION_METHODS.items():
        try:
            out[display] = load_fusion_results(key)
        except FileNotFoundError:
            print(f"  [skip] {display}: results JSON not present yet.")
    return out


# ── Per-method summary helpers ───────────────────────────────────────────────
def fold_metrics(folds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (macro_f1_array, accuracy_array) over folds."""
    return (np.array([f["macro_f1"] for f in folds]),
            np.array([f["accuracy"] for f in folds]))


def mean_per_class_f1(folds: list[dict]) -> np.ndarray:
    """Mean per-class F1 across folds. Shape (NUM_CLASSES,)."""
    return np.mean(np.stack([f["per_class_f1"] for f in folds]), axis=0)


def pooled_confusion_matrix(folds: list[dict]) -> np.ndarray:
    """Sum per-fold (non-normalized) confusion matrices into one, then
    row-normalize so each row sums to 1 (matches HARMES paper convention).
    """
    pooled = np.sum(np.stack([np.array(f["confusion_matrix"]) for f in folds]), axis=0)
    row_sums = pooled.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid /0 for any unused class
    return pooled / row_sums


# ── Common plot helpers ──────────────────────────────────────────────────────
def annotate_axis_baseline(ax, value: float, label: str, *, ls="--", color="#888"):
    """Draw a horizontal reference line with a label (e.g. AST baseline).

    Label is placed OUTSIDE the right edge of the plot area so it never
    overlaps with bars or data points.
    """
    ax.axhline(value, ls=ls, color=color, lw=0.8, zorder=0)
    ax.text(1.01, value, label,
            ha="left", va="center", color=color, fontsize=8,
            transform=ax.get_yaxis_transform(), clip_on=False)


def legend_outside_right(ax, *, title=None, **kw):
    """Place the legend vertically OUTSIDE the right edge of the plot area.

    Rule across all v3 figures: legends never overlap data — they sit in
    their own column to the right of the axes.
    """
    return ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        title=title,
        borderaxespad=0.0,
        **kw,
    )


def arrow(a: str, b: str) -> str:
    """Return a 'A → B' string using matplotlib mathtext arrow.

    The Unicode `→` glyph isn't in many fonts (renders as a tofu □ in SVGs).
    Mathtext's `\\to` is always available regardless of the system font.
    """
    return rf"{a} $\to$ {b}"


def confusion_heatmap(ax, cm: np.ndarray, *, title: str = "",
                      labels: Iterable[str] = PAPER_LABELS,
                      cmap: str = None, annot_size: int = 6):
    """Draw a confusion matrix the same way every notebook should.

    `cm` is expected to be row-normalized (values in [0, 1]).
    """
    cmap = cmap or CM_CMAP
    sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=list(labels), yticklabels=list(labels),
                cbar=False, square=True, linewidths=0.3, linecolor="white",
                annot_kws={"size": annot_size})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title:
        ax.set_title(title)


__all__ = [
    # Re-exports from src.viz
    "apply_paper_style", "FUSION_COLORS", "MODALITY_COLORS", "ACTIVITY_COLORS",
    "FOLD_COLORS", "CM_CMAP", "CM_DIFF_CMAP", "save_fig",
    # Labels
    "PAPER_LABELS", "paper_label_for_id", "ID_TO_ACTIVITY", "NUM_CLASSES",
    # Paths
    "RESULTS_DIR", "FIGURES_ROOT", "TABLES_ROOT",
    # Registry
    "FUSION_METHODS", "UNIMODAL_MODELS", "UNIMODAL_MODALITY",
    "MODEL_CSV_KEYS", "ALL_MODELS_ORDER", "LEFT_HANDERS",
    # Loaders
    "load_fusion_results", "load_unimodal_results", "load_all_fusion",
    # Helpers
    "fold_metrics", "mean_per_class_f1", "pooled_confusion_matrix",
    "annotate_axis_baseline", "confusion_heatmap",
    "legend_outside_right", "arrow",
    # Stdlib re-exports for convenience
    "np", "plt", "sns",
]
