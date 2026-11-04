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
# # Reliability Diagrams & Expected Calibration Error
#
# For each model, do its softmax confidences match its actual accuracy?
# A perfectly-calibrated model lies on the diagonal: when it says 0.8,
# it's right 80% of the time.
#
# **Expected Calibration Error (ECE):** weighted mean |confidence − accuracy|
# across 10 equal-width bins. Lower ECE = better calibrated.
#
# Data: `predictions/all_models_raw_predictions.csv.gz` (full softmax for all
# 16 classes per window).
#
# Outputs → `results/calibration/`.

# %%
from notebooks._common import *
import pandas as pd
import csv

OUT = FIGURES_ROOT / "calibration"
OUT.mkdir(parents=True, exist_ok=True)

PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]
preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv.gz",
    usecols=["model", "true_label", "pred_label"] + PROB_COLS,
)
preds["max_prob"] = preds[PROB_COLS].max(axis=1)
preds["correct"]  = (preds["true_label"] == preds["pred_label"]).astype(int)
print(f"Loaded {len(preds):,} rows; {preds['model'].nunique()} models")

# %% [markdown]
# ## 1 · Reliability curves — grid of 10 models

# %%
BINS = np.linspace(0, 1, 11)   # 10 bins

def reliability(df, bins=BINS):
    bin_idx = np.digitize(df["max_prob"], bins, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
    centers, accs, confs, counts = [], [], [], []
    for b in range(len(bins) - 1):
        sel = bin_idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        centers.append((bins[b] + bins[b + 1]) / 2)
        accs.append(df.loc[sel, "correct"].mean())
        confs.append(df.loc[sel, "max_prob"].mean())
        counts.append(n)
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)

def ece(df, bins=BINS):
    """Expected Calibration Error (lower is better)."""
    bin_idx = np.digitize(df["max_prob"], bins, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
    total = len(df)
    err = 0.0
    for b in range(len(bins) - 1):
        sel = bin_idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        acc  = df.loc[sel, "correct"].mean()
        conf = df.loc[sel, "max_prob"].mean()
        err += (n / total) * abs(acc - conf)
    return err

# Order by ECE so the best-calibrated comes first
eces = {}
for display, key in MODEL_CSV_KEYS.items():
    sub = preds[preds["model"] == key]
    eces[display] = ece(sub)
order = sorted(eces, key=lambda m: eces[m])

n_cols = 5
n_rows = int(np.ceil(len(order) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.2 * n_rows),
                         sharex=True, sharey=True)
axes = axes.flatten()
for i, display in enumerate(order):
    ax = axes[i]
    sub = preds[preds["model"] == MODEL_CSV_KEYS[display]]
    centers, accs, confs, counts = reliability(sub)
    color = FUSION_COLORS.get(display) or MODALITY_COLORS[UNIMODAL_MODALITY[display]]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#888")
    ax.plot(confs, accs, "o-", color=color, lw=1.6, markersize=4)
    # Gap shading
    ax.fill_between(confs, accs, confs, color=color, alpha=0.18,
                    linewidth=0)
    ax.set_title(f"{display}   ECE={eces[display]:.3f}", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    if i % n_cols == 0: ax.set_ylabel("Accuracy")
    if i >= len(order) - n_cols: ax.set_xlabel("Confidence")
for j in range(len(order), len(axes)):
    axes[j].axis("off")
plt.suptitle("Reliability diagrams — diagonal = perfect calibration "
             "(sorted by ECE, best first)", y=1.01)
plt.tight_layout()
save_fig(fig, OUT / "reliability_grid")

# %% [markdown]
# ## 2 · ECE ranking — single bar chart

# %%
fig, ax = plt.subplots(figsize=(9, 3.6))
xs = np.arange(len(order))
ece_vals = [eces[m] for m in order]
bar_colors = [FUSION_COLORS.get(m) or MODALITY_COLORS[UNIMODAL_MODALITY[m]]
              for m in order]
ax.bar(xs, ece_vals, color=bar_colors, edgecolor="white", linewidth=0.4)
ax.set_xticks(xs); ax.set_xticklabels(order, rotation=28, ha="right", fontsize=9)
ax.set_ylabel("Expected Calibration Error")
ax.set_title("ECE ranking — lower is better-calibrated")
for x, v in zip(xs, ece_vals):
    ax.text(x, v + 0.003, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="#222")
save_fig(fig, OUT / "ece_ranking")

# %% [markdown]
# ## 3 · Confidence distribution — correct vs wrong (per model)

# %%
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.0 * n_rows),
                         sharex=True, sharey=False)
axes = axes.flatten()
hist_bins = np.linspace(0, 1, 21)
for i, display in enumerate(order):
    ax = axes[i]
    sub = preds[preds["model"] == MODEL_CSV_KEYS[display]]
    ax.hist(sub.loc[sub["correct"] == 1, "max_prob"], bins=hist_bins,
            color=FUSION_COLORS["GMF"], alpha=0.7, label="Correct")
    ax.hist(sub.loc[sub["correct"] == 0, "max_prob"], bins=hist_bins,
            color="#E76F51", alpha=0.7, label="Wrong")
    ax.set_title(display, fontsize=9)
    if i % n_cols == 0: ax.set_ylabel("Windows")
    if i >= len(order) - n_cols: ax.set_xlabel("Top-class prob.")
for j in range(len(order), len(axes)):
    axes[j].axis("off")
axes[0].legend(loc="upper left", fontsize=7, frameon=False)
plt.suptitle("Top-class confidence — correct vs wrong", y=1.01)
plt.tight_layout()
save_fig(fig, OUT / "confidence_hist_grid")

# %% [markdown]
# ## 4 · Summary CSV

# %%
table_path = OUT / "calibration_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "ece", "mean_confidence_correct",
                "mean_confidence_wrong"])
    for display in order:
        sub = preds[preds["model"] == MODEL_CSV_KEYS[display]]
        mc_corr = sub.loc[sub["correct"] == 1, "max_prob"].mean()
        mc_wrong = sub.loc[sub["correct"] == 0, "max_prob"].mean()
        w.writerow([display, f"{eces[display]:.4f}",
                    f"{mc_corr:.4f}", f"{mc_wrong:.4f}"])
print(f"Wrote {table_path}")
print()
print("  ECE ranking (best calibrated first):")
for display in order:
    print(f"    {display:<18} ECE={eces[display]:.4f}")
