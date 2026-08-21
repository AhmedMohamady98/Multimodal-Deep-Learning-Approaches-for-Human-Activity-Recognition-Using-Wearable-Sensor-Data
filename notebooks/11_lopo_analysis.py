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
# # LOPO Analysis — GMF, 20 participants, v3 (leakage-free)
#
# Leave-One-Participant-Out cross-validation: 20 folds, each with 18 train +
# 1 val + 1 test. Matches the HARMES paper's main evaluation protocol.
# Outputs go to `results/v3/10s/lopo/`.
#
# Left-handed participants (paper Table 2): **P07, P10, P14**. We highlight
# them throughout — they are the population the HARMES paper showed audio
# rescues most.

# %%
from notebooks.v3._common import *
import json, csv
import pandas as pd

OUT = FIGURES_ROOT / "lopo"
OUT.mkdir(parents=True, exist_ok=True)

LEFT_HANDERS = {7, 10, 14}
HARMES_PAPER_BASELINE = 0.760   # IMU_L + IMU_R + Audio (5s), Table 5

results = json.loads((RESULTS_DIR / "lopo" / "lopo_results.json").read_text())
per_p = sorted(results["per_participant"], key=lambda r: r["participant"])
mean_f1, std_f1 = results["mean_macro_f1"], results["std_macro_f1"]

print(f"LOPO mean F1 = {mean_f1:.4f} ± {std_f1:.3f}  "
      f"(global pooled = {results['global_macro_f1']:.4f})")
print(f"vs HARMES paper baseline (0.760): Δ = {mean_f1 - HARMES_PAPER_BASELINE:+.4f}")

# %% [markdown]
# ## 1 · Per-participant F1 (left-handers highlighted)

# %%
participants = [r["participant"] for r in per_p]
f1s          = np.array([r["macro_f1"] for r in per_p])
colors = [FUSION_COLORS["GMF"] if p not in LEFT_HANDERS else "#E76F51"
          for p in participants]

fig, ax = plt.subplots(figsize=(11, 3.8))
bars = ax.bar(participants, f1s, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Participant")
ax.set_ylabel("Macro F1")
ax.set_title("LOPO macro F1 per participant (left-handers in terracotta)")
ax.set_ylim(0, 1.0)
ax.set_xticks(participants)
ax.set_xticklabels([f"P{p:02d}" for p in participants], rotation=0, fontsize=8)
# Baseline + LOPO mean reference lines; labels placed outside the right edge
# so they never overlap bars.
annotate_axis_baseline(ax, HARMES_PAPER_BASELINE,
                       f"HARMES paper baseline = {HARMES_PAPER_BASELINE:.3f}",
                       ls="--", color="#888")
annotate_axis_baseline(ax, mean_f1,
                       f"LOPO mean = {mean_f1:.3f}",
                       ls=":", color=FUSION_COLORS["GMF"])
# value labels
for p, v in zip(participants, f1s):
    ax.text(p, v + 0.012, f"{v:.2f}",
            ha="center", va="bottom", fontsize=6.5, color="#222")
save_fig(fig, OUT / "per_participant_f1")

# %% [markdown]
# ## 2 · Right-handers vs left-handers (gap analysis)

# %%
rh = f1s[[i for i, r in enumerate(per_p) if r["participant"] not in LEFT_HANDERS]]
lh = f1s[[i for i, r in enumerate(per_p) if r["participant"] in LEFT_HANDERS]]

fig, ax = plt.subplots(figsize=(5.5, 3.6))
groups = ["Right-handers (n=17)", "Left-handers (n=3)"]
means_g  = [rh.mean(), lh.mean()]
stds_g   = [rh.std(),  lh.std()]
g_colors = [FUSION_COLORS["GMF"], "#E76F51"]
ax.bar(groups, means_g, yerr=stds_g, color=g_colors,
       edgecolor="white", linewidth=0.6, capsize=3)
ax.set_ylabel("Macro F1")
ax.set_title("LOPO — handedness gap")
ax.set_ylim(0, 1.0)
for x, m, s in zip(range(2), means_g, stds_g):
    ax.text(x, m + s + 0.015, f"{m:.3f}",
            ha="center", va="bottom", fontsize=9, color="#222")
gap = rh.mean() - lh.mean()
ax.text(0.5, 0.97, f"Δ = {gap:+.3f}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color="#222", weight="semibold")
save_fig(fig, OUT / "handedness_gap")

# %% [markdown]
# ## 3 · Per-class F1 heatmap (participants × classes)

# %%
pc_matrix = np.stack([r["per_class_f1"] for r in per_p])   # (20, 16)

fig, ax = plt.subplots(figsize=(11.5, 6.8))
sns.heatmap(pc_matrix, ax=ax, cmap=CM_CMAP, vmin=0, vmax=1,
            xticklabels=PAPER_LABELS,
            yticklabels=[f"P{p:02d}" for p in participants],
            cbar_kws={"label": "F1"}, linewidths=0.3, linecolor="white",
            annot=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
# Highlight left-hander rows by tick label color
for tl, p in zip(ax.get_yticklabels(), participants):
    if p in LEFT_HANDERS:
        tl.set_color("#E76F51")
        tl.set_weight("semibold")
ax.set_title("Per-class F1 — every participant × every activity (left-handers in terracotta)")
ax.set_xlabel("Activity"); ax.set_ylabel("")
plt.tight_layout()
save_fig(fig, OUT / "per_class_heatmap")

# %% [markdown]
# ## 4 · Pooled confusion matrix (all 20 participants)

# %%
pooled = np.sum(np.stack([np.array(r["confusion_matrix"]) for r in per_p]), axis=0)
row_sum = pooled.sum(axis=1, keepdims=True); row_sum[row_sum == 0] = 1
cm_pooled = pooled / row_sum

fig, ax = plt.subplots(figsize=(10.5, 9.0))
confusion_heatmap(ax, cm_pooled,
                  title=f"GMF LOPO — pooled confusion matrix (all 20 participants)  —  mean macro F1 = {mean_f1:.3f}",
                  annot_size=8)
# Strengthen all text so the matrix is legible and sharp at print size.
for txt in ax.texts:                       # cell numbers
    txt.set_fontweight("black")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8.5, fontweight="black")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9.5, fontweight="black")
ax.set_xlabel("Predicted", fontsize=13, fontweight="black")
ax.set_ylabel("True",      fontsize=13, fontweight="black")
# Center the GRID itself in the figure. The long y-tick labels otherwise eat the
# left margin and push the grid right, so we lay out, then recenter the axes box
# on the figure midline and put the title at that same x = 0.5. Reserve headroom
# (rect top=0.93) so the title has clear space above the grid.
_title = ax.get_title(); ax.set_title("")
plt.tight_layout(rect=[0, 0, 1, 0.93])
_pos = ax.get_position()
ax.set_position([0.5 - _pos.width / 2, _pos.y0, _pos.width, _pos.height])
fig.suptitle(_title, x=0.5, y=0.975, fontsize=12, fontweight="black", ha="center")
save_fig(fig, OUT / "confusion_matrix_pooled")

# %% [markdown]
# ## 5 · Best vs worst participant (P06 vs P10)
#
# Quick reference here; the full deep dive lives in
# `18_best_vs_worst_participant`.

# %%
best  = max(per_p, key=lambda r: r["macro_f1"])
worst = min(per_p, key=lambda r: r["macro_f1"])
print(f"  best:  P{best['participant']:02d}  F1={best['macro_f1']:.4f}")
print(f"  worst: P{worst['participant']:02d}  F1={worst['macro_f1']:.4f}  "
      f"({'LEFT-HANDED' if worst['participant'] in LEFT_HANDERS else 'right-handed'})")

# %% [markdown]
# ## 6 · Confidence distribution from raw predictions (calibration teaser)

# %%
preds = pd.read_csv(RESULTS_DIR / "lopo" / "lopo_raw_predictions.csv")
prob_cols = [c for c in preds.columns if c.startswith("prob_class_")]
preds["max_prob"] = preds[prob_cols].max(axis=1)
preds["correct"] = preds["correct"].astype(int)

fig, ax = plt.subplots(figsize=(7, 3.4))
bins = np.linspace(0, 1, 21)
ax.hist([preds.loc[preds["correct"] == 1, "max_prob"],
         preds.loc[preds["correct"] == 0, "max_prob"]],
        bins=bins, stacked=False,
        color=[FUSION_COLORS["GMF"], "#E76F51"],
        label=["Correct", "Wrong"],
        edgecolor="white", linewidth=0.4)
ax.set_xlabel("Predicted probability of top class")
ax.set_ylabel("# Windows")
ax.set_title("LOPO — confidence histogram (correct vs wrong)")
legend_outside_right(ax, title="Window")
save_fig(fig, OUT / "confidence_histogram")

# %% [markdown]
# ## 7 · Per-class F1 bar — mean across 20 participants (ranked)
#
# Heatmap (§3) shows distribution; this bar is the one-glance summary.

# %%
pc_mean = pc_matrix.mean(axis=0)
pc_std  = pc_matrix.std(axis=0)
order_c = np.argsort(pc_mean)[::-1]
labels_c = [PAPER_LABELS[i] for i in order_c]

fig, ax = plt.subplots(figsize=(11, 4.0))
xs = np.arange(NUM_CLASSES)
ax.bar(xs, pc_mean[order_c], yerr=pc_std[order_c],
       color=FUSION_COLORS["GMF"], edgecolor="white", linewidth=0.4,
       capsize=2.5, error_kw={"elinewidth": 0.8, "ecolor": "#444"})
ax.set_xticks(xs)
ax.set_xticklabels(labels_c, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("LOPO per-class F1 (mean ± std)")
ax.set_title("LOPO per-class F1 — averaged across 20 participants (ranked)")
ax.set_ylim(0, 1.05)
for x, m, s in zip(xs, pc_mean[order_c], pc_std[order_c]):
    ax.text(x, m + s + 0.012, f"{m:.2f}",
            ha="center", va="bottom", fontsize=7, color="#222")
save_fig(fig, OUT / "lopo_perclass_bar")

# %% [markdown]
# ## 8 · Summary table

# %%
table_path = OUT / "lopo_per_participant.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["participant", "n_windows", "macro_f1", "accuracy", "handedness"])
    for r in per_p:
        w.writerow([r["participant"], r["n_windows"],
                    f"{r['macro_f1']:.4f}", f"{r['accuracy']:.4f}",
                    "left" if r["participant"] in LEFT_HANDERS else "right"])
    w.writerow([])
    accs_arr = np.array([r["accuracy"] for r in per_p])
    w.writerow(["mean", "", f"{mean_f1:.4f}", f"{accs_arr.mean():.4f}", ""])
    w.writerow(["std",  "", f"{std_f1:.4f}",  f"{accs_arr.std():.4f}", ""])
print(f"Wrote {table_path}")
