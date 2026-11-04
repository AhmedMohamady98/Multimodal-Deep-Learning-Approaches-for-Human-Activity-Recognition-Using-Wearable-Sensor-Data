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
# # Left-hander Analysis — 10 models × 20 participants
#
# Three confirmed left-handers (HARMES paper Table 2): **P07, P10, P14**.
# This notebook isolates how each modality/fusion method generalises to
# left-handed users, since the dominant-hand IMU is on the opposite wrist.
#
# Data source: `results/predictions/all_models_raw_predictions.csv.gz`
# (3-mod predictions, 437 600 rows = 43 760 windows × 10 models).
#
# Outputs → `results/lefthander/`.

# %%
from notebooks._common import *
import pandas as pd
import csv
from sklearn.metrics import f1_score

OUT = FIGURES_ROOT / "lefthander"
OUT.mkdir(parents=True, exist_ok=True)

preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv.gz",
    usecols=["participant_id", "fold", "model", "true_label",
             "pred_label", "correct"],
)
preds["is_lh"] = preds["participant_id"].isin(LEFT_HANDERS)
print(f"Loaded {len(preds):,} prediction rows; models: {preds['model'].nunique()}")

# %% [markdown]
# ## 1 · Per-method macro F1 — right-handers vs left-handers

# %%
def macro_f1(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return f1_score(df["true_label"], df["pred_label"],
                    average="macro", labels=list(range(NUM_CLASSES)),
                    zero_division=0)

# TSMixer is excluded from the gap figures. At 0.088 macro F1 it is barely
# above chance for both handedness groups, so its gap is near zero for the
# uninformative reason that the model fails equally on everyone. Sorting by
# gap would place it among the "most robust" models, which inverts what the
# figure is meant to show. It stays in the per-participant heatmaps below,
# where a near-empty column is honest context rather than a false ranking.
GAP_MODELS = {d: k for d, k in MODEL_CSV_KEYS.items() if d != "TSMixer"}

rows = []
for display, key in GAP_MODELS.items():
    m = preds[preds["model"] == key]
    rh = macro_f1(m[~m["is_lh"]])
    lh = macro_f1(m[m["is_lh"]])
    rows.append({"model": display, "rh": rh, "lh": lh, "gap": rh - lh})

# Sort by gap ascending (smallest gap = most robust to handedness)
rows.sort(key=lambda r: r["gap"])
labels = [r["model"] for r in rows]
rh_vals = [r["rh"] for r in rows]
lh_vals = [r["lh"] for r in rows]

fig, ax = plt.subplots(figsize=(10.5, 4.2))
x = np.arange(len(labels))
width = 0.4
ax.bar(x - width / 2, rh_vals, width,
       color=FUSION_COLORS["GMF"], edgecolor="white", linewidth=0.4,
       label="Right-handers (n=17)")
ax.bar(x + width / 2, lh_vals, width,
       color="#E76F51", edgecolor="white", linewidth=0.4,
       label="Left-handers (n=3)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
ax.set_ylabel("Macro F1")
ax.set_title("Right-handers vs left-handers "
             "(sorted by gap, smallest first = most robust)")
ax.set_ylim(0, 1.0)
for xi, (r, l) in enumerate(zip(rh_vals, lh_vals)):
    ax.text(xi - width / 2, r + 0.012, f"{r:.2f}",
            ha="center", va="bottom", fontsize=6.5, color="#222")
    ax.text(xi + width / 2, l + 0.012, f"{l:.2f}",
            ha="center", va="bottom", fontsize=6.5, color="#222")
legend_outside_right(ax, title="Group")
save_fig(fig, OUT / "rh_vs_lh_per_method")

# %% [markdown]
# ## 2 · Handedness gap (RH − LH) ranking
#
# Smaller bar = method generalises better to left-handers.

# %%
gaps = [r["gap"] for r in rows]
bar_colors = ["#264653" if g <= 0 else "#E76F51" for g in gaps]

fig, ax = plt.subplots(figsize=(10, 4.0))
ax.bar(range(len(rows)), gaps, color=bar_colors,
       edgecolor="white", linewidth=0.4)
ax.axhline(0, color="#222", lw=0.7)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
ax.set_ylabel("RH F1  −  LH F1")
ax.set_title("Handedness gap per model — lower is better")

# Without an explicit range matplotlib scales tightly to the bars, which
# leaves the value labels outside the axes: the negative one on AST then
# collides with the bottom spine. Pad both ends so every label has room.
lo, hi = min(gaps), max(gaps)
span = hi - lo
pad_lo = span * 0.14        # room for the label hanging below the negative bar
pad_hi = span * 0.10        # room for the label above the tallest bar
ax.set_ylim(lo - pad_lo, hi + pad_hi)

offset = span * 0.022
for xi, g in enumerate(gaps):
    va = "bottom" if g >= 0 else "top"
    ax.text(xi, g + (offset if g >= 0 else -offset),
            f"{g:+.3f}", ha="center", va=va, fontsize=7, color="#222")
save_fig(fig, OUT / "handedness_gap_ranking")

# %% [markdown]
# ## 3 · Per-participant macro F1 heatmap (LH rows highlighted)

# %%
pp_rows = []
for p in sorted(preds["participant_id"].unique()):
    for display, key in MODEL_CSV_KEYS.items():
        sub = preds[(preds["model"] == key) & (preds["participant_id"] == p)]
        pp_rows.append({"p": p, "model": display, "f1": macro_f1(sub)})

heat = (pd.DataFrame(pp_rows)
        .pivot(index="p", columns="model", values="f1")
        .reindex(columns=ALL_MODELS_ORDER))

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(heat, ax=ax, cmap=CM_CMAP, vmin=0, vmax=1,
            annot=True, fmt=".2f",
            xticklabels=heat.columns, yticklabels=[f"P{p:02d}" for p in heat.index],
            cbar_kws={"label": "Macro F1"},
            linewidths=0.3, linecolor="white",
            annot_kws={"size": 8.5, "weight": "bold"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right",
                   fontsize=11, fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, fontweight="bold")

# Highlight LH rows: terracotta row label, bold cell values, and box the best
# (green) and worst (red) model for each left-hander.
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
row_order = list(heat.index)
ncols = heat.shape[1]
GREEN, RED = "#2e7d32", "#c62828"
for tl, p in zip(ax.get_yticklabels(), heat.index):
    if p in LEFT_HANDERS:
        tl.set_color("#E76F51")
        tl.set_weight("bold")
# Humidity (TSMixer) is near-zero for everyone, so it is the trivial minimum and
# not a handedness effect; exclude it when picking the "worst" informative model.
cols = list(heat.columns)
hum_cols = [j for j, c in enumerate(cols) if "TSMixer" in c or "Humidity" in c]
for p in LEFT_HANDERS:
    if p not in row_order:
        continue
    i = row_order.index(p)
    vals = heat.loc[p].values.astype(float)
    jmax = int(np.nanargmax(vals))
    vals_inf = vals.copy()
    for j in hum_cols:
        vals_inf[j] = np.nan                     # ignore humidity for the worst
    jmin = int(np.nanargmin(vals_inf))
    ax.add_patch(Rectangle((jmax, i), 1, 1, facecolor=GREEN, alpha=0.45,
                           edgecolor=GREEN, lw=1.5, zorder=4))
    ax.add_patch(Rectangle((jmin, i), 1, 1, facecolor=RED, alpha=0.40,
                           edgecolor=RED, lw=1.5, zorder=4))
    for j in range(ncols):                       # bold the LH-row numbers
        t = ax.texts[i * ncols + j]
        t.set_fontweight("bold"); t.set_zorder(6)
    # The shaded cells get a forced white label so they read uniformly (seaborn's
    # auto-contrast would otherwise pick black on the lighter ones).
    ax.texts[i * ncols + jmax].set_color("white")
    ax.texts[i * ncols + jmin].set_color("white")

legend_handles = [
    mpatches.Patch(facecolor=GREEN, alpha=0.45, edgecolor=GREEN,
                   label="Best model for left-hander"),
    mpatches.Patch(facecolor=RED, alpha=0.40, edgecolor=RED,
                   label="Worst model for left-hander (excl. humidity)"),
]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
          frameon=True, fontsize=8, handlelength=1.4, borderpad=0.6)

ax.set_xlabel("Model"); ax.set_ylabel("")
ax.set_title("Per-participant macro F1 — every model × every participant "
             "(left-handers P07/P10/P14 highlighted)")
plt.tight_layout()
save_fig(fig, OUT / "participant_model_heatmap")

# %% [markdown]
# ## 4 · Per-class F1 — left-handers only (which activities suffer?)

# %%
def per_class_f1(df: pd.DataFrame) -> np.ndarray:
    if len(df) == 0:
        return np.zeros(NUM_CLASSES)
    return f1_score(df["true_label"], df["pred_label"],
                    average=None, labels=list(range(NUM_CLASSES)),
                    zero_division=0)

# Average per-class F1 across LH participants, per method
lh_pc = {}
for display, key in MODEL_CSV_KEYS.items():
    sub = preds[(preds["model"] == key) & (preds["is_lh"])]
    lh_pc[display] = per_class_f1(sub)

# Pick 4 representative methods (best unimodal each modality + best fusion)
focus = ["TinyHAR", "AST", "GMF", "CMA"]
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(NUM_CLASSES)
n = len(focus)
bar_w = 0.8 / n
focus_colors = {
    "TinyHAR": MODALITY_COLORS["IMU"],
    "AST":     MODALITY_COLORS["Audio"],
    "GMF":     FUSION_COLORS["GMF"],
    "CMA":     FUSION_COLORS["CMA"],
}
for i, m in enumerate(focus):
    ax.bar(x + (i - (n - 1) / 2) * bar_w, lh_pc[m], bar_w,
           color=focus_colors[m], edgecolor="white", linewidth=0.3,
           label=m)
ax.set_xticks(x)
ax.set_xticklabels(PAPER_LABELS, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Macro F1 (LH pooled)")
ax.set_title("Per-class F1 on left-handers — IMU/Audio unimodals vs best fusions")
ax.set_ylim(0, 1.0)
legend_outside_right(ax, title="Model")
save_fig(fig, OUT / "per_class_lh_focus")

# %% [markdown]
# ## 5 · How much does audio rescue left-handers?
#
# Δ(LH) = best fusion LH F1  −  TinyHAR LH F1.  Per class.

# %%
delta_audio_rescue = lh_pc["GMF"] - lh_pc["TinyHAR"]
order_dr = np.argsort(delta_audio_rescue)[::-1]
labels_dr = [PAPER_LABELS[i] for i in order_dr]
bar_colors = ["#2A9D8F" if d > 0 else "#E76F51" for d in delta_audio_rescue[order_dr]]

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.bar(range(NUM_CLASSES), delta_audio_rescue[order_dr], color=bar_colors,
       edgecolor="white", linewidth=0.4)
ax.axhline(0, color="#222", lw=0.7)
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(labels_dr, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("GMF F1  −  TinyHAR F1  (LH pooled)")
ax.set_title("Audio rescue on left-handers — per-class GMF minus IMU-only")
for xi, d in enumerate(delta_audio_rescue[order_dr]):
    va = "bottom" if d > 0 else "top"
    ax.text(xi, d + (0.01 if d > 0 else -0.01), f"{d:+.2f}",
            ha="center", va=va, fontsize=7, color="#222")
save_fig(fig, OUT / "audio_rescue_per_class")

# %% [markdown]
# ## 6 · Compact 3 × 10 heatmap — three left-handers across every model

# %%
lh_rows = []
for p in sorted(LEFT_HANDERS):
    for display, key in MODEL_CSV_KEYS.items():
        sub = preds[(preds["model"] == key) & (preds["participant_id"] == p)]
        lh_rows.append({"p": p, "model": display, "f1": macro_f1(sub)})
lh_mat = (pd.DataFrame(lh_rows)
          .pivot(index="p", columns="model", values="f1")
          .reindex(columns=ALL_MODELS_ORDER))

fig, ax = plt.subplots(figsize=(10, 2.6))
sns.heatmap(lh_mat, ax=ax, cmap=CM_CMAP, vmin=0, vmax=1,
            annot=True, fmt=".2f",
            xticklabels=lh_mat.columns,
            yticklabels=[f"P{p:02d}" for p in lh_mat.index],
            cbar_kws={"label": "Macro F1"},
            linewidths=0.4, linecolor="white",
            annot_kws={"size": 9})
ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right", fontsize=9)
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_title("Three left-handers × ten models")
plt.tight_layout()
save_fig(fig, OUT / "lefthander_heatmap_3x10")

# %% [markdown]
# ## 7 · Per-class F1 heatmap — one row per (LH participant, method)

# %%
for p in sorted(LEFT_HANDERS):
    pc_rows = []
    for display, key in MODEL_CSV_KEYS.items():
        sub = preds[(preds["model"] == key) & (preds["participant_id"] == p)]
        pc_rows.append(per_class_f1(sub))
    pc_mat = pd.DataFrame(pc_rows, index=list(MODEL_CSV_KEYS.keys()),
                          columns=PAPER_LABELS)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.heatmap(pc_mat, ax=ax, cmap=CM_CMAP, vmin=0, vmax=1,
                annot=True, fmt=".2f",
                xticklabels=pc_mat.columns, yticklabels=pc_mat.index,
                cbar_kws={"label": "F1"}, linewidths=0.3, linecolor="white",
                annot_kws={"size": 6.5})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f"P{p:02d} (left-handed) — per-class F1 by model")
    ax.set_xlabel("Activity"); ax.set_ylabel("Model")
    plt.tight_layout()
    save_fig(fig, OUT / f"p{p:02d}_per_class_heatmap")

# %% [markdown]
# ## 8 · Handedness gap — broken down by fold
#
# Are gaps consistent across folds, or driven by one bad fold?
# (P07 in fold 0; P10 + P14 in fold 1; no LH in fold 2.)

# %%
fold_rows = []
for fold in sorted(preds["fold"].unique()):
    for display, key in MODEL_CSV_KEYS.items():
        sub = preds[(preds["model"] == key) & (preds["fold"] == fold)]
        rh = macro_f1(sub[~sub["is_lh"]])
        lh = macro_f1(sub[sub["is_lh"]])
        gap = rh - lh if not np.isnan(lh) else np.nan
        fold_rows.append({"fold": fold, "model": display,
                          "rh": rh, "lh": lh, "gap": gap})
fold_df = pd.DataFrame(fold_rows)

# Only folds 0 and 1 contain left-handed participants, so fold 2 would be an
# empty third of the legend. TSMixer is dropped for the same reason it is
# dropped from the other gap figures: at chance level its gap is ~0 for a
# reason unrelated to handedness.
PANEL_ORDER = [m for m in ALL_MODELS_ORDER if m != "TSMixer"]
FOLDS_WITH_LH = [f for f in (0, 1, 2)
                 if fold_df[(fold_df["fold"] == f)]["gap"].notna().any()]

fig, ax = plt.subplots(figsize=(10, 4.0))
xs = np.arange(len(PANEL_ORDER))
width = 0.38
for slot, fi in enumerate(FOLDS_WITH_LH):
    vals = (fold_df[fold_df["fold"] == fi].set_index("model")["gap"]
            .reindex(PANEL_ORDER))
    label = f"Fold {fi}  ({'P07' if fi == 0 else 'P10 + P14'})"
    ax.bar(xs + (slot - (len(FOLDS_WITH_LH) - 1) / 2) * width, vals.values,
           width, color=FOLD_COLORS[fi], edgecolor="white", linewidth=0.4,
           label=label)
ax.axhline(0, color="#222", lw=0.7)
ax.set_xticks(xs); ax.set_xticklabels(PANEL_ORDER, rotation=28,
                                      ha="right", fontsize=9)
ax.set_ylabel("RH F1  −  LH F1  (fold-specific)")
ax.set_title("Handedness gap per fold — the gap is not stable across folds")
legend_outside_right(ax, title="Fold")
save_fig(fig, OUT / "lefthander_gap_all_folds")

fold_path = OUT / "lefthander_gap_by_fold.csv"
fold_df.to_csv(fold_path, index=False, float_format="%.4f")
print(f"Wrote {fold_path}")

# %% [markdown]
# ## 9 · Summary table

# %%
table_path = OUT / "lefthander_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "rh_macro_f1", "lh_macro_f1", "gap_rh_minus_lh"])
    for r in rows:
        w.writerow([r["model"],
                    f"{r['rh']:.4f}", f"{r['lh']:.4f}", f"{r['gap']:+.4f}"])
print(f"Wrote {table_path}")
print()
print("  Method ranking (smallest gap first):")
for r in rows:
    print(f"    {r['model']:<18} RH={r['rh']:.3f}  LH={r['lh']:.3f}   "
          f"Δ={r['gap']:+.3f}")
