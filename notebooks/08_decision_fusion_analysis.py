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
# # Decision Fusion — per-method deep dive
#
# All metrics for Decision Fusion: per-fold F1, per-class F1, pooled
# confusion matrix, per-fold confusion matrices, top confused class pairs.
# Outputs go to `results/fusion/decision_fusion/`.

# %%
from notebooks._common import *
import csv

DISPLAY_NAME = "Decision Fusion"
METHOD_KEY   = "decision_fusion"

OUT = FIGURES_ROOT / "fusion" / METHOD_KEY
OUT.mkdir(parents=True, exist_ok=True)

folds = load_fusion_results(METHOD_KEY)
COLOR = FUSION_COLORS[DISPLAY_NAME]

f1s, accs = fold_metrics(folds)
print(f"Decision Fusion: mean F1 = {f1s.mean():.4f} ± {f1s.std():.3f}, "
      f"acc = {accs.mean():.4f}")

# %% [markdown]
# ## 1 · Per-fold F1 + accuracy

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
fold_labels = ["Fold 0", "Fold 1", "Fold 2"]

# F1
axes[0].bar(fold_labels, f1s, color=COLOR, edgecolor="white", linewidth=0.6)
axes[0].set_ylabel("Macro F1")
axes[0].set_title(f"Decision Fusion — Macro F1 per fold")
axes[0].set_ylim(0, 1.0)
for x, v in enumerate(f1s):
    axes[0].text(x, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, color="#222")

# Accuracy
axes[1].bar(fold_labels, accs, color=COLOR, edgecolor="white", linewidth=0.6, alpha=0.85)
axes[1].set_ylabel("Accuracy")
axes[1].set_title(f"Decision Fusion — Accuracy per fold")
axes[1].set_ylim(0, 1.0)
for x, v in enumerate(accs):
    axes[1].text(x, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, color="#222")

plt.tight_layout()
save_fig(fig, OUT / "per_fold")

# %% [markdown]
# ## 2 · Per-class F1 (mean across folds)

# %%
mean_pc = mean_per_class_f1(folds)
order = np.argsort(mean_pc)[::-1]   # ranked high → low

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(NUM_CLASSES), mean_pc[order], color=COLOR,
       edgecolor="white", linewidth=0.4)
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels([PAPER_LABELS[i] for i in order],
                   rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean F1")
ax.set_title(f"Decision Fusion — per-class F1 (3-fold mean, ranked)")
ax.set_ylim(0, 1.0)
ax.axhline(mean_pc.mean(), ls="--", color="#888", lw=0.8, zorder=0)
ax.text(NUM_CLASSES - 0.5, mean_pc.mean() + 0.01,
        f"mean = {mean_pc.mean():.3f}",
        ha="right", va="bottom", color="#666", fontsize=8)
save_fig(fig, OUT / "per_class_f1")

# %% [markdown]
# ## 3 · Pooled confusion matrix (row-normalized, all folds combined)

# %%
cm_pooled = pooled_confusion_matrix(folds)
fig, ax = plt.subplots(figsize=(8.8, 7.8))
confusion_heatmap(ax, cm_pooled,
                  title=f"Decision Fusion — pooled confusion matrix (3-fold)",
                  annot_size=6)
plt.tight_layout()
save_fig(fig, OUT / "confusion_matrix")

# %% [markdown]
# ## 4 · Per-fold confusion matrices (side by side)

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
for ax, f, fold_name in zip(axes, folds, fold_labels):
    cm = np.array(f["confusion_matrix"], dtype=float)
    cm = cm / np.where(cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True))
    confusion_heatmap(ax, cm, title=f"{fold_name} — F1={f['macro_f1']:.3f}", annot_size=4)
plt.tight_layout()
save_fig(fig, OUT / "confusion_matrix_per_fold")

# %% [markdown]
# ## 5 · Top confused class pairs (where does this method err the most?)

# %%
cm_unnorm = np.sum(np.stack([np.array(f["confusion_matrix"]) for f in folds]), axis=0)
# Build off-diagonal confusion list
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i == j:
            continue
        count = int(cm_unnorm[i, j])
        if count > 0:
            pairs.append((PAPER_LABELS[i], PAPER_LABELS[j], count))
pairs.sort(key=lambda t: t[2], reverse=True)
top_k = 12
top_pairs = pairs[:top_k]

labels = [arrow(a, b) for a, b, _ in top_pairs]
counts = [c for _, _, c in top_pairs]
fig, ax = plt.subplots(figsize=(8.5, 4.5))
y = np.arange(len(top_pairs))[::-1]
ax.barh(y, counts, color=COLOR, edgecolor="white", linewidth=0.4)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Window count (3-fold sum)")
ax.set_title(f"Decision Fusion — top {top_k} confused class pairs")
for yi, c in zip(y, counts):
    ax.text(c + max(counts) * 0.01, yi, str(c),
            va="center", fontsize=8, color="#222")
save_fig(fig, OUT / "top_confused_pairs")

# %% [markdown]
# ## 6 · Per-fold summary table

# %%
table_path = OUT / f"decision_fusion_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold", "test_participants", "n_windows", "macro_f1", "accuracy"])
    for fd in folds:
        w.writerow([fd["fold"],
                    "P" + "/P".join(str(p) for p in fd["test_participants"]),
                    fd["n_windows"],
                    f"{fd['macro_f1']:.4f}",
                    f"{fd['accuracy']:.4f}"])
    w.writerow([])
    w.writerow(["mean", "", "", f"{f1s.mean():.4f}", f"{accs.mean():.4f}"])
    w.writerow(["std",  "", "", f"{f1s.std():.4f}",  f"{accs.std():.4f}"])
print(f"Wrote {table_path}")
