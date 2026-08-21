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
# # Unimodal Baselines — v3 (leakage-free, 16 classes incl. Null)
#
# Compares the three single-modality encoders evaluated on the same 3-fold
# protocol used for fusion. Outputs go to `results/v3/10s/unimodal/`.
#
# Each model is paired with its modality color (paper Fig 5 convention):
# IMU = warm yellow, Audio = green-teal, Humidity = soft blue.

# %%
from notebooks.v3._common import *

OUT = FIGURES_ROOT / "unimodal"
OUT.mkdir(parents=True, exist_ok=True)

uni = load_unimodal_results()
print("Models:", list(uni.keys()))

# %% [markdown]
# ## 1 · Mean F1 + accuracy across folds

# %%
summary = []
for model in UNIMODAL_MODELS:
    f1s, accs = fold_metrics(uni[model])
    summary.append({"model": model,
                    "modality": UNIMODAL_MODALITY[model],
                    "mean_f1": f1s.mean(), "std_f1": f1s.std(),
                    "mean_acc": accs.mean(), "std_acc": accs.std()})

# Sort by F1 descending for the bar chart
summary.sort(key=lambda r: r["mean_f1"], reverse=True)
models   = [r["model"] for r in summary]
means    = [r["mean_f1"] for r in summary]
stds     = [r["std_f1"]  for r in summary]
colors   = [MODALITY_COLORS[UNIMODAL_MODALITY[m]] for m in models]

fig, ax = plt.subplots(figsize=(5.5, 3.3))
ax.bar(range(len(models)), means, yerr=stds, color=colors,
       edgecolor="white", linewidth=0.6, capsize=2.8,
       error_kw={"elinewidth": 0.9, "ecolor": "#444"})
ax.set_xticks(range(len(models)))
ax.set_xticklabels([f"{m}\n({UNIMODAL_MODALITY[m]})" for m in models])
ax.set_ylabel("Macro F1")
ax.set_title("Unimodal baselines — 3-fold CV macro F1")
ax.set_ylim(0, 0.85)
for x, m, s in zip(range(len(models)), means, stds):
    ax.text(x, m + s + 0.012, f"{m:.3f}",
            ha="center", va="bottom", fontsize=8, color="#222")
save_fig(fig, OUT / "leaderboard")

# %% [markdown]
# ## 2 · Per-fold breakdown (which modality is most stable?)

# %%
fig, ax = plt.subplots(figsize=(6, 3.3))
x = np.arange(len(models))
width = 0.26
for fi, fold_name in enumerate(["Fold 0", "Fold 1", "Fold 2"]):
    fold_f1s = [uni[m][fi]["macro_f1"] for m in models]
    ax.bar(x + (fi - 1) * width, fold_f1s, width,
           color=FOLD_COLORS[fi], edgecolor="white", linewidth=0.4,
           label=fold_name)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Macro F1")
ax.set_title("Per-fold F1 by unimodal encoder")
ax.set_ylim(0, 0.85)
legend_outside_right(ax, title="Fold")
save_fig(fig, OUT / "per_fold_breakdown")

# %% [markdown]
# ## 3 · Per-class F1 across all 3 unimodal encoders

# %%
mean_pc = {m: mean_per_class_f1(uni[m]) for m in UNIMODAL_MODELS}
fig, ax = plt.subplots(figsize=(11, 4.0))
x = np.arange(NUM_CLASSES)
n = len(UNIMODAL_MODELS)
bar_w = 0.8 / n
for i, m in enumerate(UNIMODAL_MODELS):
    ax.bar(x + (i - (n - 1) / 2) * bar_w, mean_pc[m], bar_w,
           color=MODALITY_COLORS[UNIMODAL_MODALITY[m]],
           edgecolor="white", linewidth=0.3,
           label=f"{m} ({UNIMODAL_MODALITY[m]})")
ax.set_xticks(x)
ax.set_xticklabels(PAPER_LABELS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean F1 (3-fold)")
ax.set_title("Per-class F1 — unimodal encoders")
ax.set_ylim(0, 1.0)
legend_outside_right(ax, title="Encoder")
save_fig(fig, OUT / "per_class_f1")

# %% [markdown]
# ## 4 · Confusion matrices (one per encoder)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, model in zip(axes, UNIMODAL_MODELS):
    cm = pooled_confusion_matrix(uni[model])
    confusion_heatmap(ax, cm,
                      title=f"{model} ({UNIMODAL_MODALITY[model]})",
                      annot_size=5)
plt.tight_layout()
save_fig(fig, OUT / "confusion_matrices")

# %% [markdown]
# ## 5 · Separate confusion matrices per encoder (individual colour per modality)
#
# IMU → Blues, Audio → Oranges, Humidity → Greens  (matches bar-chart modality colours).
# Each figure is saved as cm_tinyhar, cm_ast, cm_tsmixer.

# %%
UNIMODAL_CMAPS = {
    "TinyHAR": "Blues",
    "AST":     "Oranges",
    "TSMixer": "Greens",
}

for model in UNIMODAL_MODELS:
    cm   = pooled_confusion_matrix(uni[model])
    f1   = fold_metrics(uni[model])[0].mean()
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    confusion_heatmap(
        ax, cm,
        title=f"{model} ({UNIMODAL_MODALITY[model]})  —  macro F1 = {f1:.3f}",
        cmap=UNIMODAL_CMAPS[model],
        annot_size=9,
    )
    # Strengthen all text so the matrix is legible at print size.
    for txt in ax.texts:                       # cell numbers
        txt.set_fontweight("bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, fontweight="black")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True",      fontsize=12, fontweight="bold")
    ax.title.set_fontweight("bold")
    plt.tight_layout()
    save_fig(fig, OUT / f"cm_{model.lower()}")
    plt.close(fig)
    print(f"  Saved  results/v3/10s/unimodal/cm_{model.lower()}.pdf")

# %% [markdown]
# ## 6 · Summary table

# %%
import csv
table_path = OUT / "unimodal_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "modality", "mean_f1", "std_f1", "mean_acc", "std_acc"])
    for r in summary:
        w.writerow([r["model"], r["modality"],
                    f"{r['mean_f1']:.4f}", f"{r['std_f1']:.4f}",
                    f"{r['mean_acc']:.4f}", f"{r['std_acc']:.4f}"])
print(f"Wrote {table_path}")
print()
for r in summary:
    print(f"  {r['model']:<10} ({r['modality']:<8}) "
          f"F1={r['mean_f1']:.4f} ± {r['std_f1']:.3f}   acc={r['mean_acc']:.4f}")
