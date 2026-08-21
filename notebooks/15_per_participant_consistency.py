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
# # Per-Participant Consistency — how stable is each method across people?
#
# Three views of the same underlying matrix (20 participants × 10 models):
#
# 1. **Heatmap** — every participant × every model, F1 in cell.
# 2. **Bars** — per-method mean ± std across participants (low std = consistent).
# 3. **Violin** — full F1 distribution per method (shape reveals outliers).
#
# Source: `predictions/all_models_raw_predictions.csv`.
# Outputs → `results/v3/10s/per_participant_consistency/`.

# %%
from notebooks.v3._common import *
import pandas as pd
import csv
from sklearn.metrics import f1_score

OUT = FIGURES_ROOT / "per_participant_consistency"
OUT.mkdir(parents=True, exist_ok=True)

preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv",
    usecols=["participant_id", "model", "true_label", "pred_label"],
)

def macro_f1(df):
    if len(df) == 0:
        return float("nan")
    return f1_score(df["true_label"], df["pred_label"],
                    average="macro", labels=list(range(NUM_CLASSES)),
                    zero_division=0)

# Build (participant × model) F1 matrix
rows = []
for p in sorted(preds["participant_id"].unique()):
    for display, key in MODEL_CSV_KEYS.items():
        sub = preds[(preds["model"] == key) & (preds["participant_id"] == p)]
        rows.append({"participant": p, "model": display, "f1": macro_f1(sub)})
mat = (pd.DataFrame(rows)
       .pivot(index="participant", columns="model", values="f1")
       .reindex(columns=ALL_MODELS_ORDER))
print(f"Matrix: {mat.shape[0]} participants × {mat.shape[1]} models")

# %% [markdown]
# ## 1 · Heatmap — every participant × every model

# %%
fig, ax = plt.subplots(figsize=(10, 7.2))
sns.heatmap(mat, ax=ax, cmap=CM_CMAP, vmin=0, vmax=1,
            annot=True, fmt=".2f",
            xticklabels=mat.columns,
            yticklabels=[f"P{p:02d}" for p in mat.index],
            cbar_kws={"label": "Macro F1"},
            linewidths=0.3, linecolor="white",
            annot_kws={"size": 6.5})
ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right", fontsize=9)
for tl, p in zip(ax.get_yticklabels(), mat.index):
    if p in LEFT_HANDERS:
        tl.set_color("#E76F51")
        tl.set_weight("semibold")
ax.set_xlabel("Model"); ax.set_ylabel("")
ax.set_title("Per-participant macro F1 — every participant × every model "
             "(left-handers in terracotta)")
plt.tight_layout()
save_fig(fig, OUT / "per_participant_heatmap")

# %% [markdown]
# ## 2 · Bars — mean ± std across participants (lower std = more consistent)

# %%
means = mat.mean(axis=0)
stds  = mat.std(axis=0)
order = means.sort_values(ascending=False).index
means_o = means[order]; stds_o = stds[order]

bar_colors = [FUSION_COLORS.get(m) or MODALITY_COLORS[UNIMODAL_MODALITY[m]]
              for m in order]

fig, ax = plt.subplots(figsize=(11, 4.0))
xs = np.arange(len(order))
ax.bar(xs, means_o, yerr=stds_o, color=bar_colors,
       edgecolor="white", linewidth=0.5, capsize=2.8,
       error_kw={"elinewidth": 0.9, "ecolor": "#444"})
ax.set_xticks(xs); ax.set_xticklabels(order, rotation=28, ha="right", fontsize=9)
ax.set_ylabel("Macro F1")
ax.set_title("Per-participant consistency — mean ± std across 20 participants")
ax.set_ylim(0, 1.0)
for x, m, s in zip(xs, means_o, stds_o):
    ax.text(x, m + s + 0.012, f"{m:.3f}\n±{s:.3f}",
            ha="center", va="bottom", fontsize=6.5, color="#222")
save_fig(fig, OUT / "per_participant_consistency_bars")

# %% [markdown]
# ## 3 · Violin — full F1 distribution per method (reveals outliers)

# %%
long = mat.reset_index().melt(id_vars="participant", var_name="model",
                              value_name="f1")
# Preserve order
long["model"] = pd.Categorical(long["model"], categories=order, ordered=True)
long = long.sort_values("model")

fig, ax = plt.subplots(figsize=(11.5, 4.6))
sns.violinplot(data=long, x="model", y="f1", ax=ax,
               palette=[FUSION_COLORS.get(m) or MODALITY_COLORS[UNIMODAL_MODALITY[m]]
                        for m in order],
               inner=None, linewidth=0.8, cut=0,
               hue="model", legend=False)
# Overlay individual points
sns.stripplot(data=long, x="model", y="f1", ax=ax,
              color="#222", alpha=0.55, size=2.6, jitter=0.18)
# Highlight left-handers with terracotta dots
lh_long = long[long["participant"].isin(LEFT_HANDERS)]
sns.stripplot(data=lh_long, x="model", y="f1", ax=ax,
              color="#E76F51", alpha=0.95, size=4.0, jitter=0.18,
              edgecolor="white", linewidth=0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right", fontsize=9)
ax.set_ylabel("Macro F1")
ax.set_xlabel("")
ax.set_title("Per-participant F1 distribution — violin per model "
             "(terracotta dots = left-handers)")
ax.set_ylim(0, 1.0)
save_fig(fig, OUT / "per_participant_consistency_violin")

# %% [markdown]
# ## 4 · Summary CSV

# %%
table_path = OUT / "per_participant_consistency.csv"
mat.to_csv(table_path, float_format="%.4f")
print(f"Wrote {table_path}")

stats_path = OUT / "consistency_stats.csv"
with open(stats_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "mean_f1", "std_f1", "min_f1", "max_f1", "range"])
    for m in order:
        col = mat[m]
        w.writerow([m, f"{col.mean():.4f}", f"{col.std():.4f}",
                    f"{col.min():.4f}", f"{col.max():.4f}",
                    f"{col.max() - col.min():.4f}"])
print(f"Wrote {stats_path}")
print()
print("  Consistency ranking (mean F1 desc):")
for m in order:
    col = mat[m]
    print(f"    {m:<18} mean={col.mean():.3f}  std={col.std():.3f}  "
          f"range={col.max() - col.min():.3f}")
