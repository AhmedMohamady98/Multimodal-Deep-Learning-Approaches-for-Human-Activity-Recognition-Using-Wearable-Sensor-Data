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
# # Confused Class Pairs — Cross-method
#
# Which class pairs are universally confused — that is, every model
# struggles with them? And which pairs are confused only by a specific
# model? Counts are pooled across all 3 folds.
#
# Outputs → `results/confusion/`.

# %%
from notebooks._common import *
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix

OUT = FIGURES_ROOT / "confusion"
OUT.mkdir(parents=True, exist_ok=True)

preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv.gz",
    usecols=["model", "true_label", "pred_label"],
)
print(f"Loaded {len(preds):,} rows; {preds['model'].nunique()} models")

def cm_for(model_key):
    sub = preds[preds["model"] == model_key]
    return confusion_matrix(sub["true_label"], sub["pred_label"],
                            labels=list(range(NUM_CLASSES)))

# Per-method off-diagonal confusion (row-normalized) → averaged across methods
def offdiag_norm(cm):
    cm = cm.astype(float)
    s = cm.sum(axis=1, keepdims=True); s[s == 0] = 1
    cm_n = cm / s
    np.fill_diagonal(cm_n, 0.0)
    return cm_n

all_cms = {display: offdiag_norm(cm_for(key))
           for display, key in MODEL_CSV_KEYS.items()}
mean_cm = np.mean(np.stack(list(all_cms.values())), axis=0)

# %% [markdown]
# ## 1 · Mean off-diagonal confusion heatmap (averaged across 10 models)

# %%
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(mean_cm, ax=ax, cmap=CM_CMAP, vmin=0, vmax=mean_cm.max(),
            annot=True, fmt=".2f",
            xticklabels=PAPER_LABELS, yticklabels=PAPER_LABELS,
            cbar_kws={"label": "Mean off-diag rate"},
            linewidths=0.3, linecolor="white",
            annot_kws={"size": 5})
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Universal confusion — mean off-diagonal rate across all 10 models")
plt.tight_layout()
save_fig(fig, OUT / "mean_offdiag_heatmap")

# %% [markdown]
# ## 2 · Top-K universal confused pairs

# %%
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i == j: continue
        if mean_cm[i, j] > 0:
            pairs.append((PAPER_LABELS[i], PAPER_LABELS[j], mean_cm[i, j]))
pairs.sort(key=lambda t: t[2], reverse=True)
TOP_K = 15
top_pairs = pairs[:TOP_K]

labels = [arrow(a, b) for a, b, _ in top_pairs]
rates  = [r for _, _, r in top_pairs]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(top_pairs))[::-1]
ax.barh(y, rates, color="#264653", edgecolor="white", linewidth=0.4)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Mean off-diagonal confusion rate (across 10 models)")
ax.set_title(f"Top-{TOP_K} universally-confused class pairs")
for yi, r in zip(y, rates):
    ax.text(r + max(rates) * 0.01, yi, f"{r:.3f}",
            va="center", fontsize=8, color="#222")
save_fig(fig, OUT / "top_universal_pairs")

# %% [markdown]
# ## 3 · Per-method ranking of the top-5 universal pairs

# %%
top5_pairs = pairs[:5]

# TSMixer is left out of this panel. At F1 0.088 it sits barely above
# chance, so its off-diagonal mass reflects whichever classes it happens
# to over-predict rather than a genuine confusion between them, and its
# bars are an order of magnitude out of line with every other model's.
# The pair *selection* above is unchanged and still uses all ten models.
PANEL_MODELS = [m for m in MODEL_CSV_KEYS if m != "TSMixer"]

fig, ax = plt.subplots(figsize=(9, 4.4))
xs = np.arange(len(PANEL_MODELS))
width = 0.8 / len(top5_pairs)
pair_colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
for i, (a, b, _) in enumerate(top5_pairs):
    ai = PAPER_LABELS.index(a); bj = PAPER_LABELS.index(b)
    vals = [all_cms[m][ai, bj] for m in PANEL_MODELS]
    ax.bar(xs + (i - (len(top5_pairs) - 1) / 2) * width, vals, width,
           color=pair_colors[i], edgecolor="white", linewidth=0.3,
           label=arrow(a, b))
ax.set_xticks(xs)
ax.set_xticklabels(PANEL_MODELS, rotation=28, ha="right", fontsize=9)
ax.set_ylabel("Off-diagonal rate")
ax.set_title("Top-5 universal confused pairs — per model")

# Legend below the axes rather than outside-right: this figure now runs at
# \linewidth in the main text, where a right-hand legend column would
# squeeze the plotting area to roughly half the text width.
handles, pair_labels = ax.get_legend_handles_labels()
fig.legend(handles, pair_labels, loc="upper center",
           bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False,
           fontsize=8.5, title="Confused pair", title_fontsize=9)
fig.subplots_adjust(bottom=0.30)
save_fig(fig, OUT / "top5_pairs_per_method")

# %% [markdown]
# ## 4 · Summary CSV

# %%
table_path = OUT / "confused_pairs.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true_class", "pred_class", "mean_offdiag_rate"]
               + list(MODEL_CSV_KEYS.keys()))
    for a, b, rate in pairs[:30]:
        ai = PAPER_LABELS.index(a); bj = PAPER_LABELS.index(b)
        row = [a, b, f"{rate:.4f}"] + [
            f"{all_cms[m][ai, bj]:.4f}" for m in MODEL_CSV_KEYS]
        w.writerow(row)
print(f"Wrote {table_path}")
print(f"\n  Top-10 universally-confused pairs:")
for a, b, r in pairs[:10]:
    print(f"    {a:<24} -> {b:<24}  {r:.4f}")
