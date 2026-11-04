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
# # Modality Ablation — IMU + Audio only (drop humidity)
#
# 3-modality vs 2-modality comparison for the three best fusion methods
# (Late Fusion, GMF, CMA). Answers: *does humidity actually contribute,
# or are all the gains coming from IMU + Audio?*
#
# Data sources:
#   - `predictions/all_models_raw_predictions.csv.gz`       (3-modality)
#   - `predictions/2mod_all_models_raw_predictions.csv.gz`  (IMU + Audio)
#
# Outputs → `results/ablation/`.

# %%
from notebooks._common import *
import pandas as pd
import csv
from sklearn.metrics import f1_score, confusion_matrix

OUT = FIGURES_ROOT / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

ABLATED = ["Late Fusion", "GMF", "CMA"]   # methods that have a 2-mod variant

p3 = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv.gz",
    usecols=["participant_id", "fold", "model", "true_label", "pred_label"],
)
p2 = pd.read_csv(
    RESULTS_DIR / "predictions" / "2mod_all_models_raw_predictions.csv.gz",
    usecols=["participant_id", "fold", "model", "true_label", "pred_label"],
)
print(f"3-mod rows: {len(p3):,}  ·  2-mod rows: {len(p2):,}")

def macro_f1(df):
    if len(df) == 0:
        return float("nan")
    return f1_score(df["true_label"], df["pred_label"],
                    average="macro", labels=list(range(NUM_CLASSES)),
                    zero_division=0)

def per_class(df):
    return f1_score(df["true_label"], df["pred_label"],
                    average=None, labels=list(range(NUM_CLASSES)),
                    zero_division=0)

# %% [markdown]
# ## 1 · Macro F1 — 3-mod vs 2-mod (per fusion method)

# %%
rows = []
for m in ABLATED:
    k3 = MODEL_CSV_KEYS[m]
    k2 = f"{k3}_2mod"
    f3 = macro_f1(p3[p3["model"] == k3])
    f2 = macro_f1(p2[p2["model"] == k2])
    rows.append({"method": m, "f1_3mod": f3, "f1_2mod": f2,
                 "delta_hum": f3 - f2})

fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(len(rows))
width = 0.36
ax.bar(x - width / 2, [r["f1_3mod"] for r in rows], width,
       color=FUSION_COLORS["GMF"], edgecolor="white", linewidth=0.5,
       label="IMU + Audio + Humidity")
ax.bar(x + width / 2, [r["f1_2mod"] for r in rows], width,
       color="#E9C46A", edgecolor="white", linewidth=0.5,
       label="IMU + Audio only")
ax.set_xticks(x); ax.set_xticklabels([r["method"] for r in rows])
ax.set_ylabel("Macro F1")
ax.set_title("Modality ablation — does humidity actually help?")
ax.set_ylim(0, 1.0)
for xi, r in enumerate(rows):
    ax.text(xi - width / 2, r["f1_3mod"] + 0.012, f"{r['f1_3mod']:.3f}",
            ha="center", va="bottom", fontsize=8, color="#222")
    ax.text(xi + width / 2, r["f1_2mod"] + 0.012, f"{r['f1_2mod']:.3f}",
            ha="center", va="bottom", fontsize=8, color="#222")
legend_outside_right(ax, title="Modalities")
save_fig(fig, OUT / "macro_f1_3mod_vs_2mod")

# %% [markdown]
# ## 2 · Humidity contribution Δ (3-mod − 2-mod) — per class

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, r, m in zip(axes, rows, ABLATED):
    k3 = MODEL_CSV_KEYS[m]; k2 = f"{k3}_2mod"
    pc3 = per_class(p3[p3["model"] == k3])
    pc2 = per_class(p2[p2["model"] == k2])
    delta = pc3 - pc2
    order = np.argsort(delta)[::-1]
    bar_colors = ["#2A9D8F" if d > 0 else "#E76F51" for d in delta[order]]
    ax.bar(range(NUM_CLASSES), delta[order], color=bar_colors,
           edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#222", lw=0.7)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([PAPER_LABELS[i] for i in order],
                       rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{m}  (Δ={r['delta_hum']:+.3f})")
    ax.axhline(0, lw=0.6, color="#444")
axes[0].set_ylabel("F1 (3-mod) − F1 (2-mod)")
plt.suptitle("Per-class humidity contribution — green = humidity helps, "
             "red = it hurts", y=1.02)
plt.tight_layout()
save_fig(fig, OUT / "per_class_humidity_delta")

# %% [markdown]
# ## 3 · Humidity water-activity deep dive
#
# The hypothesis behind including humidity: it should detect water-involving
# activities (washing, watering, etc.) better than IMU/audio alone. Here we
# isolate just those classes and check whether the 3-mod variant pulls them up.

# %%
WATER_ACTIVITIES = [
    "Washing hands", "Washing dishes", "Watering plants",
    "Cleaning out dishwasher", "Cleaning table",
    "Disinfecting hands", "Apply hand cream", "Making tea",
]
water_idx = [PAPER_LABELS.index(a) for a in WATER_ACTIVITIES
             if a in PAPER_LABELS]

water_rows = []
for m in ABLATED:
    k3 = MODEL_CSV_KEYS[m]; k2 = f"{k3}_2mod"
    pc3 = per_class(p3[p3["model"] == k3])
    pc2 = per_class(p2[p2["model"] == k2])
    for cls_i in water_idx:
        water_rows.append({
            "method": m, "activity": PAPER_LABELS[cls_i],
            "f1_3mod": pc3[cls_i], "f1_2mod": pc2[cls_i],
            "delta": pc3[cls_i] - pc2[cls_i],
        })

water_df = pd.DataFrame(water_rows)

# Heatmap: methods × water activities, color = Δ humidity contribution
heat = water_df.pivot(index="activity", columns="method", values="delta")[ABLATED]
heat = heat.reindex(WATER_ACTIVITIES)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
vmax = max(abs(heat.values.min()), abs(heat.values.max()))
sns.heatmap(heat, ax=ax, cmap=CM_DIFF_CMAP, center=0,
            vmin=-vmax, vmax=vmax,
            annot=True, fmt="+.2f",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "F1 (3-mod) − F1 (2-mod)"},
            annot_kws={"size": 9})
# CM_DIFF_CMAP is RdBu_r, so high values are red and low values are blue.
# Delta is F1(3-mod) minus F1(2-mod), so a positive delta means humidity
# helped, and that maps to red.
ax.set_title("Humidity contribution — water-related activities only\n"
             "red = humidity helps, blue = it hurts")
ax.set_xlabel("Fusion method"); ax.set_ylabel("")
plt.tight_layout()
save_fig(fig, OUT / "humidity_water_activity_deep_dive")

# Bar chart: mean Δ across the 3 methods, per water activity
mean_delta_water = water_df.groupby("activity")["delta"].mean()
mean_delta_water = mean_delta_water.reindex(WATER_ACTIVITIES)
order_w = mean_delta_water.sort_values(ascending=False).index
bar_colors = ["#2A9D8F" if mean_delta_water[a] > 0 else "#E76F51"
              for a in order_w]

fig, ax = plt.subplots(figsize=(8, 3.6))
xs = np.arange(len(order_w))
ax.bar(xs, mean_delta_water[order_w].values, color=bar_colors,
       edgecolor="white", linewidth=0.4)
ax.axhline(0, color="#222", lw=0.7)
ax.set_xticks(xs)
ax.set_xticklabels(order_w, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean Δ across 3 fusion methods")
ax.set_title("Mean humidity contribution per water activity")
for x, v in zip(xs, mean_delta_water[order_w].values):
    va = "bottom" if v >= 0 else "top"
    ax.text(x, v + (0.005 if v >= 0 else -0.005), f"{v:+.3f}",
            ha="center", va=va, fontsize=8, color="#222")
save_fig(fig, OUT / "humidity_contribution")

water_path = OUT / "humidity_water_activity.csv"
water_df.to_csv(water_path, index=False, float_format="%.4f")
print(f"Wrote {water_path}")

# %% [markdown]
# ## 4 · Summary table

# %%
table_path = OUT / "ablation_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "f1_3mod", "f1_2mod", "delta_humidity"])
    for r in rows:
        w.writerow([r["method"],
                    f"{r['f1_3mod']:.4f}",
                    f"{r['f1_2mod']:.4f}",
                    f"{r['delta_hum']:+.4f}"])
print(f"Wrote {table_path}")
print()
print("  Humidity contribution per method:")
for r in rows:
    print(f"    {r['method']:<14} 3-mod={r['f1_3mod']:.4f}  "
          f"2-mod={r['f1_2mod']:.4f}   Δ={r['delta_hum']:+.4f}")
mean_delta = sum(r["delta_hum"] for r in rows) / len(rows)
print(f"\n  Mean humidity contribution: {mean_delta:+.4f}")
