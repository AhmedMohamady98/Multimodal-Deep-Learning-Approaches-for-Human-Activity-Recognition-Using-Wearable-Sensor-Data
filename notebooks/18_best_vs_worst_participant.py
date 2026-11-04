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
# # Best vs Worst Participant — LOPO Deep Dive
#
# Picks the highest- and lowest-scoring participants from the LOPO run and
# compares them head-to-head: per-class F1 deltas, confusion-matrix diffs,
# and the activities where the gap is widest.
#
# Outputs → `results/lefthander/best_vs_worst/`.

# %%
from notebooks._common import *
import json, csv

LEFT_HANDERS = {7, 10, 14}

OUT = FIGURES_ROOT / "lefthander" / "best_vs_worst"
OUT.mkdir(parents=True, exist_ok=True)

results = json.loads((RESULTS_DIR / "lopo" / "lopo_results.json").read_text())
per_p = {r["participant"]: r for r in results["per_participant"]}

best_p  = max(per_p.values(), key=lambda r: r["macro_f1"])
worst_p = min(per_p.values(), key=lambda r: r["macro_f1"])
B, W = best_p["participant"], worst_p["participant"]
print(f"Best  participant: P{B:02d}  F1={best_p['macro_f1']:.4f}  "
      f"({'LEFT-HANDED' if B in LEFT_HANDERS else 'right-handed'})")
print(f"Worst participant: P{W:02d}  F1={worst_p['macro_f1']:.4f}  "
      f"({'LEFT-HANDED' if W in LEFT_HANDERS else 'right-handed'})")

# %% [markdown]
# ## 1 · Per-class F1 — side-by-side bars

# %%
best_pc  = np.array(best_p["per_class_f1"])
worst_pc = np.array(worst_p["per_class_f1"])

# Sort by best participant's score so the gap pattern stands out
order = np.argsort(best_pc)[::-1]
labels = [PAPER_LABELS[i] for i in order]

fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(NUM_CLASSES)
width = 0.4
ax.bar(x - width / 2, best_pc[order], width,
       color=FUSION_COLORS["GMF"], edgecolor="white", linewidth=0.4,
       label=f"P{B:02d} (best)")
ax.bar(x + width / 2, worst_pc[order], width,
       color="#E76F51", edgecolor="white", linewidth=0.4,
       label=f"P{W:02d} (worst)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("F1")
ax.set_title(f"Per-class F1 — best (P{B:02d}) vs worst (P{W:02d}) participant")
ax.set_ylim(0, 1.05)
legend_outside_right(ax, title="Participant")
save_fig(fig, OUT / "per_class_f1")

# %% [markdown]
# ## 2 · Per-class delta (best − worst) — where is the gap widest?

# %%
delta = best_pc - worst_pc
order_d = np.argsort(delta)[::-1]
labels_d = [PAPER_LABELS[i] for i in order_d]
bar_colors = ["#264653" if d > 0 else "#E76F51" for d in delta[order_d]]

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.bar(range(NUM_CLASSES), delta[order_d], color=bar_colors,
       edgecolor="white", linewidth=0.4)
ax.axhline(0, color="#222", lw=0.7)
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(labels_d, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel(f"P{B:02d} F1  −  P{W:02d} F1")
ax.set_title(f"Per-class F1 gap — P{B:02d} minus P{W:02d}")
for x, d in zip(range(NUM_CLASSES), delta[order_d]):
    va = "bottom" if d > 0 else "top"
    ax.text(x, d + (0.012 if d > 0 else -0.012), f"{d:+.2f}",
            ha="center", va=va, fontsize=7, color="#222")
save_fig(fig, OUT / "per_class_delta")

# %% [markdown]
# ## 3 · Confusion matrices side-by-side

# %%
def row_normalize(cm):
    cm = np.array(cm, dtype=float)
    s = cm.sum(axis=1, keepdims=True); s[s == 0] = 1
    return cm / s

best_cm  = row_normalize(best_p["confusion_matrix"])
worst_cm = row_normalize(worst_p["confusion_matrix"])

fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
confusion_heatmap(axes[0], best_cm,
                  title=f"P{B:02d} (best, F1={best_p['macro_f1']:.3f})",
                  annot_size=5)
confusion_heatmap(axes[1], worst_cm,
                  title=f"P{W:02d} (worst, F1={worst_p['macro_f1']:.3f})"
                        f" — {'LEFT-HANDED' if W in LEFT_HANDERS else 'right-handed'}",
                  annot_size=5)
plt.tight_layout()
save_fig(fig, OUT / "confusion_matrices")

# %% [markdown]
# ## 4 · Confusion matrix difference (best − worst) — divergent cmap

# %%
diff = best_cm - worst_cm
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(diff, ax=ax, cmap=CM_DIFF_CMAP, center=0, vmin=-0.6, vmax=0.6,
            annot=True, fmt="+.2f",
            xticklabels=PAPER_LABELS, yticklabels=PAPER_LABELS,
            cbar=False, square=True, linewidths=0.3, linecolor="white",
            annot_kws={"size": 5})
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_title(f"Confusion matrix difference  (P{B:02d}  −  P{W:02d})")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
save_fig(fig, OUT / "confusion_difference")

# %% [markdown]
# ## 5 · Top gap activities — summary table

# %%
table_path = OUT / "top_gap_activities.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["activity", "F1_best", "F1_worst", "delta"])
    for i in order_d:
        w.writerow([PAPER_LABELS[i],
                    f"{best_pc[i]:.4f}",
                    f"{worst_pc[i]:.4f}",
                    f"{delta[i]:+.4f}"])
print(f"Wrote {table_path}")
print()
print(f"  Top 5 widest gaps (P{B:02d} − P{W:02d}):")
for i in order_d[:5]:
    print(f"    {PAPER_LABELS[i]:<26} {best_pc[i]:.3f}  vs  {worst_pc[i]:.3f}   Δ={delta[i]:+.3f}")
