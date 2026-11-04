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
# # Top-K Accuracy — how often is the true class in the top-K predictions?
#
# Top-1 = standard accuracy. Top-2 / Top-3 reveal how close the model gets
# when it misclassifies. A model with low Top-1 but high Top-3 is "almost
# right" — useful for downstream re-ranking or human-in-the-loop systems.
#
# Outputs → `results/topk/`.

# %%
from notebooks._common import *
import pandas as pd
import csv

OUT = FIGURES_ROOT / "topk"
OUT.mkdir(parents=True, exist_ok=True)

PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]
preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv.gz",
    usecols=["model", "true_label"] + PROB_COLS,
)
print(f"Loaded {len(preds):,} rows; {preds['model'].nunique()} models")

# %% [markdown]
# ## 1 · Top-K table for K ∈ {1, 2, 3, 5}

# %%
K_VALUES = [1, 2, 3, 5]

results_rows = []
for display, key in MODEL_CSV_KEYS.items():
    sub = preds[preds["model"] == key]
    probs = sub[PROB_COLS].to_numpy()
    true  = sub["true_label"].to_numpy()
    # argsort descending → ranks
    ranks = np.argsort(-probs, axis=1)
    row = {"model": display}
    for k in K_VALUES:
        topk = ranks[:, :k]
        correct = (topk == true[:, None]).any(axis=1)
        row[f"top{k}"] = correct.mean()
    results_rows.append(row)

# Sort by top-1 desc
results_rows.sort(key=lambda r: r["top1"], reverse=True)

# Grouped bar chart
fig, ax = plt.subplots(figsize=(12, 4.0))
x = np.arange(len(results_rows))
nk = len(K_VALUES)
bar_w = 0.8 / nk
k_colors = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"]
for i, k in enumerate(K_VALUES):
    vals = [r[f"top{k}"] for r in results_rows]
    ax.bar(x + (i - (nk - 1) / 2) * bar_w, vals, bar_w,
           color=k_colors[i], edgecolor="white", linewidth=0.3,
           label=f"Top-{k}")
ax.set_xticks(x)
ax.set_xticklabels([r["model"] for r in results_rows], rotation=28, ha="right")
ax.set_ylabel("Accuracy")
ax.set_title("Top-K accuracy — sorted by Top-1")
ax.set_ylim(0, 1.02)
for i, k in enumerate(K_VALUES):
    for xi, r in enumerate(results_rows):
        ax.text(xi + (i - (nk - 1) / 2) * bar_w, r[f"top{k}"] + 0.005,
                f"{r[f'top{k}']:.2f}", ha="center", va="bottom",
                fontsize=6, color="#222")
legend_outside_right(ax, title="K")
save_fig(fig, OUT / "topk_grouped")

# %% [markdown]
# ## 2 · Top-1 → Top-3 gain (how much "almost right"?)

# %%
gains = [(r["model"], r["top3"] - r["top1"]) for r in results_rows]
gains.sort(key=lambda t: t[1], reverse=True)

fig, ax = plt.subplots(figsize=(9, 3.6))
xs = np.arange(len(gains))
ax.bar(xs, [g[1] for g in gains], color=FUSION_COLORS["GMF"],
       edgecolor="white", linewidth=0.4)
ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in gains],
                                       rotation=28, ha="right", fontsize=9)
ax.set_ylabel("Top-3 accuracy  −  Top-1 accuracy")
ax.set_title("How much does Top-3 rescue each model from Top-1 mistakes?")
for x, (m, v) in zip(xs, gains):
    ax.text(x, v + 0.004, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="#222")
save_fig(fig, OUT / "top1_to_top3_gain")

# %% [markdown]
# ## 3 · Summary table

# %%
table_path = OUT / "topk_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model"] + [f"top{k}" for k in K_VALUES])
    for r in results_rows:
        w.writerow([r["model"]] + [f"{r[f'top{k}']:.4f}" for k in K_VALUES])
print(f"Wrote {table_path}")
print()
hdr = "  " + " " * 18 + "  ".join(f"Top-{k}" for k in K_VALUES)
print(hdr)
for r in results_rows:
    print(f"  {r['model']:<18} " + "  ".join(
        f"{r[f'top{k}']:.4f}" for k in K_VALUES))
