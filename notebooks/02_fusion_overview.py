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
# # Fusion Overview — v3 (leakage-free, 16 classes incl. Null)
#
# All 7 fusion methods evaluated on the 3-fold protocol. Numbers and figures
# match the v3 thesis chapter. Activity labels follow the HARMES dataset
# paper Table 3 wording (e.g. "Apply hand cream", "Cleaning out dishwasher",
# "Brushing Teeth").
#
# Outputs go to `results/v3/10s/fusion/overview/`.

# %%
from notebooks.v3._common import *

OUT = FIGURES_ROOT / "fusion" / "overview"
OUT.mkdir(parents=True, exist_ok=True)

uni  = load_unimodal_results()
fus  = load_all_fusion()
ast_mean = fold_metrics(uni["AST"])[0].mean()
print(f"AST mean F1 (best unimodal baseline) = {ast_mean:.4f}")
print(f"Fusion methods loaded: {list(fus.keys())}")

# %% [markdown]
# ## 1 · Leaderboard — all 7 methods ranked by macro F1

# %%
rows = []
for name, folds in fus.items():
    f1s, accs = fold_metrics(folds)
    rows.append({"method": name, "mean_f1": f1s.mean(), "std_f1": f1s.std(),
                 "mean_acc": accs.mean(), "std_acc": accs.std(),
                 "delta_vs_ast": f1s.mean() - ast_mean})
rows.sort(key=lambda r: r["mean_f1"], reverse=True)

methods = [r["method"] for r in rows]
means   = [r["mean_f1"] for r in rows]
stds    = [r["std_f1"]  for r in rows]
colors  = [FUSION_COLORS[m] for m in methods]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
bars = ax.bar(range(len(methods)), means, yerr=stds, color=colors,
              edgecolor="white", linewidth=0.6, capsize=2.8,
              error_kw={"elinewidth": 0.9, "ecolor": "#444"})
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=28, ha="right")
ax.set_ylabel("Macro F1")
ax.set_title("Fusion methods — 3-fold CV macro F1")
ax.set_ylim(0.6, max(means) + 0.04)

# AST baseline reference — shown as a legend entry (upper right) rather than
# an inline label, so the plot stays full width.
ax.axhline(ast_mean, ls="--", color="#888", lw=1.0, zorder=0,
           label=f"AST baseline (best unimodal) = {ast_mean:.3f}")
ax.legend(loc="upper right", frameon=True, fontsize=8, framealpha=0.95)

# Label each bar with its value
for x, m, s in zip(range(len(methods)), means, stds):
    ax.text(x, m + s + 0.005, f"{m:.3f}",
            ha="center", va="bottom", fontsize=8, color="#222")

save_fig(fig, OUT / "leaderboard")

# %% [markdown]
# ## 2 · Per-fold breakdown (which methods are most stable across folds?)

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.5))
x = np.arange(len(methods))
width = 0.26
for fi, fold_name in enumerate(["Fold 0", "Fold 1", "Fold 2"]):
    fold_f1s = [fus[m][fi]["macro_f1"] for m in methods]
    ax.bar(x + (fi - 1) * width, fold_f1s, width,
           color=FOLD_COLORS[fi], edgecolor="white", linewidth=0.4,
           label=fold_name)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=28, ha="right")
ax.set_ylabel("Macro F1")
ax.set_title("Per-fold F1 by fusion method")
ax.set_ylim(0.5, 0.92)
legend_outside_right(ax, title="Fold")
save_fig(fig, OUT / "per_fold_breakdown")

# %% [markdown]
# ## 3 · Delta vs best unimodal (AST) — does fusion actually help?

# %%
deltas = [r["delta_vs_ast"] for r in rows]
fig, ax = plt.subplots(figsize=(6.5, 3.3))
bar_colors = [FUSION_COLORS[m] if d > 0 else "#bbbbbb" for m, d in zip(methods, deltas)]
ax.bar(range(len(methods)), deltas, color=bar_colors,
       edgecolor="white", linewidth=0.6)
ax.axhline(0, color="#222", lw=0.8)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=28, ha="right")
ax.set_ylabel("Δ Macro F1 vs AST")
ax.set_title("Fusion improvement over best unimodal (AST)")
for x, d in zip(range(len(methods)), deltas):
    va = "bottom" if d > 0 else "top"
    ax.text(x, d + (0.005 if d > 0 else -0.005), f"{d:+.3f}",
            ha="center", va=va, fontsize=7.5, color="#222")
save_fig(fig, OUT / "delta_vs_ast")

# %% [markdown]
# ## 4 · Per-class F1 across all methods (which activities benefit from fusion?)

# %%
mean_pc = {m: mean_per_class_f1(fus[m]) for m in methods}
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(NUM_CLASSES)
n = len(methods)
bar_w = 0.8 / n
for i, m in enumerate(methods):
    ax.bar(x + (i - (n - 1) / 2) * bar_w, mean_pc[m], bar_w,
           color=FUSION_COLORS[m], edgecolor="white", linewidth=0.3, label=m)
ax.set_xticks(x)
ax.set_xticklabels(PAPER_LABELS, rotation=45, ha="right", fontsize=9, fontweight="black")
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("black")
ax.set_ylabel("Mean F1 (3-fold)", fontsize=12, fontweight="black")
ax.set_title("Per-class F1 — all fusion methods", fontsize=13, fontweight="black")
ax.set_ylim(0, 1.0)
leg = legend_outside_right(ax, title="Method")
if leg is not None:
    leg.set_title("Method", prop={"weight": "black"})
    for txt in leg.get_texts():
        txt.set_fontweight("black")
save_fig(fig, OUT / "per_class_all_methods")

# %% [markdown]
# ## 5 · Cross-method confusion matrix grid (one CM per method, side by side)

# %%
n = len(methods)
n_cols = 4
n_rows = int(np.ceil(n / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))
axes = axes.flatten()
for i, m in enumerate(methods):
    cm = pooled_confusion_matrix(fus[m])
    confusion_heatmap(axes[i], cm, title=m, annot_size=4)
for j in range(len(methods), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
save_fig(fig, OUT / "confusion_matrix_grid")

# %% [markdown]
# ## 6 · Confusion matrix difference — best fusion (GMF) minus best unimodal (AST)
#
# Where exactly does fusion fix the unimodal model? Blue cells = GMF
# beats AST (e.g. diagonal entries fusion got right that AST missed);
# red cells = GMF newly confuses something AST handled.

# %%
cm_gmf = pooled_confusion_matrix(fus["GMF"])
cm_ast = pooled_confusion_matrix(uni["AST"])
diff = cm_gmf - cm_ast

fig, ax = plt.subplots(figsize=(9, 8))
vmax = max(abs(diff.min()), abs(diff.max()))
sns.heatmap(diff, ax=ax, cmap=CM_DIFF_CMAP, center=0,
            vmin=-vmax, vmax=vmax,
            annot=True, fmt="+.2f",
            xticklabels=PAPER_LABELS, yticklabels=PAPER_LABELS,
            cbar=False,
            linewidths=0.3, linecolor="white", square=True,
            annot_kws={"size": 8, "weight": "black"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                   fontsize=9, fontweight="black")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   fontsize=10, fontweight="bold")
ax.set_title("Confusion-matrix difference:  GMF (best fusion) − AST (best unimodal)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
ax.set_ylabel("True",      fontsize=12, fontweight="bold")

# Compact legend instead of a full colour bar.
import matplotlib.patches as mpatches
_cmap = plt.get_cmap(CM_DIFF_CMAP)
legend_handles = [
    mpatches.Patch(color=_cmap(0.88), label="Red = GMF (fusion) higher"),
    mpatches.Patch(color=_cmap(0.12), label="Blue = AST (best unimodal) higher"),
]
ax.legend(handles=legend_handles, loc="upper left",
          bbox_to_anchor=(1.01, 1.0), ncol=1, frameon=True, fontsize=8,
          handlelength=1.0, handleheight=1.0, borderpad=0.4, labelspacing=0.3,
          framealpha=0.95)

fig.subplots_adjust(bottom=0.24, top=0.94, right=0.99)
save_fig(fig, OUT / "confusion_matrix_diff_gmf_vs_ast")

# %% [markdown]
# ## 6b · Confusion matrix difference — GMF minus TinyHAR (best IMU)
#
# Same view against the IMU-only model: where does fusion fix what
# wrist motion alone cannot resolve?

# %%
cm_tinyhar = pooled_confusion_matrix(uni["TinyHAR"])
diff_t = cm_gmf - cm_tinyhar

fig, ax = plt.subplots(figsize=(9, 8))
vmax_t = max(abs(diff_t.min()), abs(diff_t.max()))
sns.heatmap(diff_t, ax=ax, cmap=CM_DIFF_CMAP, center=0,
            vmin=-vmax_t, vmax=vmax_t,
            annot=True, fmt="+.2f",
            xticklabels=PAPER_LABELS, yticklabels=PAPER_LABELS,
            cbar=False,
            linewidths=0.3, linecolor="white", square=True,
            annot_kws={"size": 8, "weight": "black"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                   fontsize=9, fontweight="black")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   fontsize=10, fontweight="bold")
ax.set_title("Confusion-matrix difference:  GMF (best fusion) − TinyHAR (IMU)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
ax.set_ylabel("True",      fontsize=12, fontweight="bold")

legend_handles_t = [
    mpatches.Patch(color=_cmap(0.88), label="Red = GMF (fusion) higher"),
    mpatches.Patch(color=_cmap(0.12), label="Blue = TinyHAR (IMU) higher"),
]
ax.legend(handles=legend_handles_t, loc="upper left",
          bbox_to_anchor=(1.01, 1.0), ncol=1, frameon=True, fontsize=8,
          handlelength=1.0, handleheight=1.0, borderpad=0.4, labelspacing=0.3,
          framealpha=0.95)

fig.subplots_adjust(bottom=0.24, top=0.94, right=0.99)
save_fig(fig, OUT / "confusion_matrix_diff_gmf_vs_tinyhar")

# %% [markdown]
# ## 7 · Summary table (CSV — for thesis)

# %%
import csv
table_path = OUT / "fusion_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "mean_f1", "std_f1", "mean_acc", "std_acc",
                "delta_vs_ast"])
    for r in rows:
        w.writerow([r["method"], f"{r['mean_f1']:.4f}", f"{r['std_f1']:.4f}",
                    f"{r['mean_acc']:.4f}", f"{r['std_acc']:.4f}",
                    f"{r['delta_vs_ast']:+.4f}"])
print(f"Wrote {table_path}")
print()
for r in rows:
    print(f"  {r['method']:<18} {r['mean_f1']:.4f} ± {r['std_f1']:.3f}   "
          f"acc={r['mean_acc']:.4f}   Δ(AST)={r['delta_vs_ast']:+.4f}")
