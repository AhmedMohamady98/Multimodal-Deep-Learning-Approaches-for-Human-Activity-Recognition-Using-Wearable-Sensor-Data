"""
Display-name mapping aligned with the HARMES dataset paper (Burchard et al.
2026, arXiv:2605.02596) Table 3.

The codebase's `ACTIVITY_TO_ID` in `src/data/dataset.py` is the source of
truth for the integer label IDs (DO NOT change — every trained checkpoint
is tied to that mapping). This module only changes the *display strings*
used in figures, tables, and report-facing CSVs, so that everything we
publish matches the dataset paper verbatim.

Three names differ from the codebase:
  codebase                paper
  ----------------------  --------------------------
  Creaming hands          Apply hand cream
  Cleaning dishwasher     Cleaning out dishwasher
  Brushing teeth          Brushing Teeth   (capital T)

All other names match. The 16th class ("Null", ID=15) is paper-aligned
(the paper calls it the "null" class in section 3.2; we keep the
sentence-cased "Null" for figure legends).

Typical usage (in notebooks / figure scripts):

    from src.data.dataset import ID_TO_ACTIVITY
    from src.data.paper_names import to_paper_name, paper_label_for_id

    paper_labels = [paper_label_for_id(i) for i in range(16)]
    plt.xticks(range(16), paper_labels, rotation=45)
"""

from src.data.dataset import ACTIVITY_TO_ID, ID_TO_ACTIVITY


# ── codebase → paper display string ──────────────────────────────────────────
CODE_TO_PAPER: dict[str, str] = {
    "Washing hands":        "Washing hands",
    "Floor cleaning":       "Floor cleaning",
    "Window cleaning":      "Window cleaning",
    "Vacuum cleaning":      "Vacuum cleaning",
    "Brushing teeth":       "Brushing Teeth",          # paper capitalizes T
    "Washing dishes":       "Washing dishes",
    "Putting away dishes":  "Putting away dishes",
    "Making tea":           "Making tea",
    "Cutting vegetables":   "Cutting vegetables",
    "Creaming hands":       "Apply hand cream",        # paper wording
    "Drinking":             "Drinking",
    "Watering plants":      "Watering plants",
    "Cleaning table":       "Cleaning table",
    "Disinfecting hands":   "Disinfecting hands",
    "Cleaning dishwasher":  "Cleaning out dishwasher", # paper wording
    "Null":                 "Null",                    # paper lowercases
}

# Sanity: every codebase activity has a paper mapping
_missing = set(ACTIVITY_TO_ID) - set(CODE_TO_PAPER)
if _missing:
    raise RuntimeError(
        f"paper_names.py is missing paper-name mappings for: {sorted(_missing)}. "
        "Update CODE_TO_PAPER above to keep ACTIVITY_TO_ID and the paper Table 3 in sync."
    )


# ── public helpers ───────────────────────────────────────────────────────────
def to_paper_name(codebase_name: str) -> str:
    """Return the paper-aligned display string for a codebase activity name.

    Raises KeyError if the codebase name is not recognized — this catches
    typos at call-site rather than producing silent wrong labels.
    """
    return CODE_TO_PAPER[codebase_name]


def paper_label_for_id(label_id: int) -> str:
    """Map a class ID (0..NUM_CLASSES-1) to the paper-aligned display string."""
    return CODE_TO_PAPER[ID_TO_ACTIVITY[label_id]]


# Pre-computed ordered list for matplotlib tick labels etc.
PAPER_LABELS: list[str] = [paper_label_for_id(i) for i in range(len(ID_TO_ACTIVITY))]


if __name__ == "__main__":
    # Quick smoke test: print the side-by-side mapping
    print(f"{'ID':>3}  {'codebase':<25}  paper")
    print("-" * 60)
    for i, code in sorted(ID_TO_ACTIVITY.items()):
        paper = CODE_TO_PAPER[code]
        diff  = "  ← renamed" if code != paper else ""
        print(f"{i:>3}  {code:<25}  {paper}{diff}")
