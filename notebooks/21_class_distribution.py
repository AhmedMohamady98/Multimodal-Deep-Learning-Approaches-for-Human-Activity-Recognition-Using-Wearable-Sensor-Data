# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (.venv)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dataset Class Distribution — v3 (16 classes incl. Null)
#
# Derived from the windowed evaluation set: how many 10 s windows fall into
# each activity class? Crucial for v3 because the windowing fix added the
# **Null** (background / transition) class — and Null is the most common
# label by a wide margin, which explains a lot of the "Activity → Null"
# confusion patterns we see in `confusion/`.
#
# Outputs → `results/v3/10s/dataset/`.

# %%
from notebooks.v3._common import *
import pandas as pd
import csv

OUT = FIGURES_ROOT / "dataset"
OUT.mkdir(parents=True, exist_ok=True)

# Use any one model's predictions — true labels are identical across models.
preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv",
    usecols=["model", "participant_id", "true_label"],
)
preds = preds[preds["model"] == MODEL_CSV_KEYS["GMF"]]   # 43 760 unique windows
print(f"Total windows: {len(preds):,}  "
      f"across {preds['participant_id'].nunique()} participants")

counts = preds["true_label"].value_counts().sort_index()
counts = counts.reindex(range(NUM_CLASSES), fill_value=0)
shares = counts / counts.sum()

# %% [markdown]
# ## 1 · Class counts — ranked

# %%
order = counts.sort_values(ascending=False).index
labels_o = [PAPER_LABELS[i] for i in order]
counts_o = counts.loc[order].values
shares_o = shares.loc[order].values

null_color = "#888"
def bar_color(i):
    return null_color if PAPER_LABELS[i] == "Null" else FUSION_COLORS["GMF"]
bar_colors = [bar_color(i) for i in order]

fig, ax = plt.subplots(figsize=(11.5, 4.2))
xs = np.arange(NUM_CLASSES)
ax.bar(xs, counts_o, color=bar_colors,
       edgecolor="white", linewidth=0.4)
ax.set_xticks(xs)
ax.set_xticklabels(labels_o, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("# windows (10 s)")
ax.set_title("Class distribution — 3-fold evaluation pool (16 classes incl. Null)")
for x, c, s in zip(xs, counts_o, shares_o):
    ax.text(x, c + max(counts_o) * 0.01, f"{c:,}\n({s * 100:.1f}%)",
            ha="center", va="bottom", fontsize=6.5, color="#222")
save_fig(fig, OUT / "class_distribution")

# %% [markdown]
# ## 2 · Class share by participant (stacked bar)
#
# Reveals how unevenly classes are distributed across people — relevant
# for LOPO, where one held-out participant may not even contain certain
# classes (the macro-F1 then treats those as "0 / 0" → counted as 0).

# %%
pp_counts = (preds.groupby(["participant_id", "true_label"])
             .size().unstack(fill_value=0)
             .reindex(columns=range(NUM_CLASSES), fill_value=0))
pp_shares = pp_counts.div(pp_counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(12, 4.6))
bottom = np.zeros(len(pp_shares))
for cls in range(NUM_CLASSES):
    label = PAPER_LABELS[cls]
    ax.bar(pp_shares.index, pp_shares[cls].values, bottom=bottom,
           color=ACTIVITY_COLORS[label], edgecolor="white", linewidth=0.2,
           label=label)
    bottom += pp_shares[cls].values
ax.set_xticks(pp_shares.index)
ax.set_xticklabels([f"P{p:02d}" for p in pp_shares.index],
                   rotation=0, fontsize=7.5)
ax.set_ylabel("Share of participant's windows")
ax.set_title("Class share per participant — stacked")
ax.set_ylim(0, 1.0)
legend_outside_right(ax, title="Activity", ncol=1, fontsize=7)
save_fig(fig, OUT / "class_share_per_participant")

# %% [markdown]
# ## 3 · Summary table

# %%
table_path = OUT / "class_distribution.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class_id", "class_name", "n_windows", "share"])
    for i in range(NUM_CLASSES):
        w.writerow([i, PAPER_LABELS[i], int(counts.loc[i]),
                    f"{shares.loc[i]:.4f}"])
print(f"Wrote {table_path}")
print()
print(f"  Total windows: {counts.sum():,}")
print(f"  Null share: {shares.loc[PAPER_LABELS.index('Null')]:.1%}")
print(f"  Smallest non-Null class: "
      f"{PAPER_LABELS[counts.drop(PAPER_LABELS.index('Null')).idxmin()]} "
      f"({counts.drop(PAPER_LABELS.index('Null')).min()} windows)")
