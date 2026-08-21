"""Visualization helpers — palette, style, save utilities aligned with
the HARMES dataset paper (Burchard et al. 2026)."""

from .paper_style import (
    apply_paper_style,
    ACTIVITY_COLORS,
    MODALITY_COLORS,
    FUSION_COLORS,
    FOLD_COLORS,
    CM_CMAP,
    CM_DIFF_CMAP,
    save_fig,
    activity_color,
)

__all__ = [
    "apply_paper_style",
    "ACTIVITY_COLORS",
    "MODALITY_COLORS",
    "FUSION_COLORS",
    "FOLD_COLORS",
    "CM_CMAP",
    "CM_DIFF_CMAP",
    "save_fig",
    "activity_color",
]
