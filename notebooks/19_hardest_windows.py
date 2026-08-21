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
# # Hardest Windows — which 10 s segments fool every model?
#
# A "hard" window is one where many models get it wrong **and** are confident
# in the wrong answer. We rank windows by a "wrongness score" pooled across
# all 10 models and visualise:
#   - which activities dominate the hard set
#   - which participants contribute most
#   - which classes the hard windows get predicted as
#
# Outputs → `results/v3/10s/hardest_windows/`.

# %%
from notebooks.v3._common import *
import pandas as pd
import csv

OUT = FIGURES_ROOT / "hardest_windows"
OUT.mkdir(parents=True, exist_ok=True)

PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]
preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv",
    usecols=["participant_id", "fold", "window_idx", "model",
             "true_label", "pred_label"] + PROB_COLS,
)
preds["wrong"] = (preds["true_label"] != preds["pred_label"]).astype(int)
preds["max_prob"] = preds[PROB_COLS].max(axis=1)
preds["confident_wrong"] = preds["wrong"] * preds["max_prob"]
print(f"Loaded {len(preds):,} rows")

# %% [markdown]
# ## 1 · Score each window: how many models wronged it, with what confidence?

# %%
KEY = ["participant_id", "fold", "window_idx"]
hard = (preds
        .groupby(KEY + ["true_label"], as_index=False)
        .agg(n_models_wrong=("wrong", "sum"),
             mean_confident_wrong=("confident_wrong", "mean")))
hard["hardness"] = hard["n_models_wrong"] + hard["mean_confident_wrong"]
hard = hard.sort_values("hardness", ascending=False).reset_index(drop=True)

print(f"  Hardest window: P{int(hard.iloc[0]['participant_id']):02d}, "
      f"fold {int(hard.iloc[0]['fold'])}, window {int(hard.iloc[0]['window_idx'])} "
      f"(true class: {PAPER_LABELS[int(hard.iloc[0]['true_label'])]}) "
      f"— {int(hard.iloc[0]['n_models_wrong'])}/10 models wrong, "
      f"mean confident-wrong = {hard.iloc[0]['mean_confident_wrong']:.3f}")

# Take top 5% as "hardest"
N_HARD = max(50, int(len(hard) * 0.05))
hardest = hard.head(N_HARD)
print(f"  Treating top-{N_HARD} windows (5% tail) as 'hard'")

# %% [markdown]
# ## 2 · Activity distribution of hard windows vs full set

# %%
all_counts = hard["true_label"].value_counts(normalize=True).sort_index()
hard_counts = hardest["true_label"].value_counts(normalize=True).sort_index()
# Align both to NUM_CLASSES
def align(s):
    return np.array([s.get(i, 0.0) for i in range(NUM_CLASSES)])
all_v  = align(all_counts)
hard_v = align(hard_counts)

fig, ax = plt.subplots(figsize=(11, 4.0))
x = np.arange(NUM_CLASSES)
width = 0.4
ax.bar(x - width / 2, all_v, width,
       color="#264653", edgecolor="white", linewidth=0.4,
       label="All windows")
ax.bar(x + width / 2, hard_v, width,
       color="#E76F51", edgecolor="white", linewidth=0.4,
       label="Hardest 5%")
ax.set_xticks(x)
ax.set_xticklabels(PAPER_LABELS, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Share of windows")
ax.set_title("Activity distribution — full set vs hardest 5%")
legend_outside_right(ax, title="Set")
save_fig(fig, OUT / "activity_distribution")

# %% [markdown]
# ## 3 · Hardness over-representation — which classes are disproportionately hard?

# %%
ratio = np.divide(hard_v, all_v, out=np.zeros_like(hard_v),
                  where=(all_v > 0))
order = np.argsort(ratio)[::-1]

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.bar(range(NUM_CLASSES), ratio[order], color="#E76F51",
       edgecolor="white", linewidth=0.4)
ax.axhline(1, color="#222", lw=0.7, ls="--")
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels([PAPER_LABELS[i] for i in order],
                   rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("hard share / overall share")
ax.set_title("Over-representation in the hardest 5% (1.0 = even, >1 = disproportionately hard)")
for xi, v in enumerate(ratio[order]):
    ax.text(xi, v + 0.05, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7, color="#222")
save_fig(fig, OUT / "hardness_overrepresentation")

# %% [markdown]
# ## 4 · Participant contribution to hard windows

# %%
pp_share_all = (hard["participant_id"].value_counts(normalize=True)
                .reindex(range(1, 21), fill_value=0).sort_index())
pp_share_hard = (hardest["participant_id"].value_counts(normalize=True)
                 .reindex(range(1, 21), fill_value=0).sort_index())

fig, ax = plt.subplots(figsize=(11, 3.6))
x = np.arange(1, 21)
width = 0.4
colors_all = ["#264653" if p not in LEFT_HANDERS else "#888"
              for p in x]
colors_hard = ["#E76F51" if p not in LEFT_HANDERS else "#8E7AB5"
               for p in x]
ax.bar(x - width / 2, pp_share_all.values, width,
       color=colors_all, edgecolor="white", linewidth=0.4,
       label="All windows")
ax.bar(x + width / 2, pp_share_hard.values, width,
       color=colors_hard, edgecolor="white", linewidth=0.4,
       label="Hardest 5%")
ax.set_xticks(x); ax.set_xticklabels([f"P{p:02d}" for p in x],
                                       rotation=0, fontsize=7)
ax.set_ylabel("Share of windows")
ax.set_title("Per-participant share — full vs hardest 5% "
             "(left-handers in lighter shade)")
legend_outside_right(ax, title="Set")
save_fig(fig, OUT / "participant_contribution")

# %% [markdown]
# ## 5 · Gallery — top 30 hardest windows table

# %%
table_path = OUT / "hardest_windows_top30.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "participant", "fold", "window_idx", "true_class",
                "n_models_wrong", "mean_confident_wrong", "hardness_score"])
    for rank, row in enumerate(hardest.head(30).itertuples(), start=1):
        w.writerow([rank, int(row.participant_id), int(row.fold),
                    int(row.window_idx),
                    PAPER_LABELS[int(row.true_label)],
                    int(row.n_models_wrong),
                    f"{row.mean_confident_wrong:.4f}",
                    f"{row.hardness:.4f}"])
print(f"Wrote {table_path}")
print()
print("  Top 10 hardest windows:")
for rank, row in enumerate(hardest.head(10).itertuples(), start=1):
    print(f"    #{rank:2d}  P{int(row.participant_id):02d}  fold{int(row.fold)}  "
          f"win {int(row.window_idx):4d}  "
          f"true={PAPER_LABELS[int(row.true_label)]:<22}  "
          f"{int(row.n_models_wrong)}/10 wrong  "
          f"conf-wrong={row.mean_confident_wrong:.3f}")
