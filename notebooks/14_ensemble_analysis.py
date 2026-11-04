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
# # Ensemble Analysis — does averaging fusion models beat any single one?
#
# Build a simple late-stage ensemble by averaging the softmax distributions
# of the K best-performing fusion methods, per window, and check whether
# the ensemble beats the single best model (GMF).
#
# Outputs → `results/ensemble/`.

# %%
from notebooks._common import *
import pandas as pd
import csv
from sklearn.metrics import f1_score

OUT = FIGURES_ROOT / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)

PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]
preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv.gz",
    usecols=["model", "participant_id", "fold", "window_idx",
             "true_label"] + PROB_COLS,
)
print(f"Loaded {len(preds):,} rows; {preds['model'].nunique()} models")

def macro_f1(true, pred):
    return f1_score(true, pred, average="macro",
                    labels=list(range(NUM_CLASSES)), zero_division=0)

# Individual model F1 (for reference)
single_f1 = {}
for display, key in MODEL_CSV_KEYS.items():
    sub = preds[preds["model"] == key]
    p = sub[PROB_COLS].to_numpy().argmax(axis=1)
    single_f1[display] = macro_f1(sub["true_label"], p)
print("Single-model F1:")
for m, v in sorted(single_f1.items(), key=lambda t: -t[1]):
    print(f"  {m:<18} {v:.4f}")

# %% [markdown]
# ## 1 · Build a fusion-only ensemble — average softmax of top-K fusions

# %%
FUSION_DISPLAYS = list(FUSION_METHODS.keys())   # 7 fusion methods
fusion_ranked = sorted([(m, single_f1[m]) for m in FUSION_DISPLAYS],
                       key=lambda t: -t[1])

# Pivot probabilities so each window has one row per model
# Use (participant, fold, window_idx) as the join key
key_cols = ["participant_id", "fold", "window_idx"]
ensembles = {}

# Pre-cache numpy probability arrays per model in row-aligned order
canonical = (preds[preds["model"] == MODEL_CSV_KEYS["GMF"]]
             .sort_values(key_cols)
             .reset_index(drop=True))
true = canonical["true_label"].to_numpy()
probs_by_model: dict[str, np.ndarray] = {}
for display, key in MODEL_CSV_KEYS.items():
    sub = (preds[preds["model"] == key]
           .sort_values(key_cols)
           .reset_index(drop=True))
    assert (sub["participant_id"].to_numpy() == canonical["participant_id"].to_numpy()).all()
    probs_by_model[display] = sub[PROB_COLS].to_numpy()

ens_results = []
for K in range(2, len(FUSION_DISPLAYS) + 1):
    selected = [m for m, _ in fusion_ranked[:K]]
    probs = np.mean(np.stack([probs_by_model[m] for m in selected]), axis=0)
    pred = probs.argmax(axis=1)
    f1 = macro_f1(true, pred)
    ens_results.append({"K": K, "members": selected, "f1": f1})

# Also: ensemble of ALL 10 models (incl. unimodals) — does humidity drag it down?
all10 = np.mean(np.stack([probs_by_model[m] for m in MODEL_CSV_KEYS]), axis=0)
ens_results.append({"K": 10, "members": list(MODEL_CSV_KEYS),
                    "f1": macro_f1(true, all10.argmax(axis=1)),
                    "label": "all 10 (incl. unimodals)"})

# Plot
fig, ax = plt.subplots(figsize=(8, 3.8))
xs = [r["K"] for r in ens_results]
ys = [r["f1"] for r in ens_results]
labels_x = []
for r in ens_results:
    if "label" in r:
        labels_x.append(r["label"])
    else:
        labels_x.append(f"top-{r['K']} fusions")
bar_colors = ["#264653"] * (len(ens_results) - 1) + ["#E76F51"]
ax.bar(range(len(ens_results)), ys, color=bar_colors,
       edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(ens_results)))
ax.set_xticklabels(labels_x, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Macro F1")
ax.set_title("Ensemble macro F1 — averaging softmax of top-K fusions")
ax.set_ylim(0.7, max(ys) + 0.04)

best_single = max(single_f1.values())
best_single_name = max(single_f1, key=single_f1.get)
annotate_axis_baseline(ax, best_single,
                       f"best single ({best_single_name}) = {best_single:.3f}",
                       ls=":", color=FUSION_COLORS["GMF"])
for x, v in zip(range(len(ens_results)), ys):
    ax.text(x, v + 0.005, f"{v:.4f}",
            ha="center", va="bottom", fontsize=7.5, color="#222")
save_fig(fig, OUT / "ensemble_topk")

# %% [markdown]
# ## 2 · Weighted ensemble — weight each model by its own F1²

# %%
weights = np.array([single_f1[m] ** 2 for m in FUSION_DISPLAYS])
weights = weights / weights.sum()
weighted = np.zeros_like(probs_by_model[FUSION_DISPLAYS[0]])
for w, m in zip(weights, FUSION_DISPLAYS):
    weighted += w * probs_by_model[m]
f1_weighted = macro_f1(true, weighted.argmax(axis=1))
print(f"\n  Weighted (F1²) ensemble of 7 fusions: F1 = {f1_weighted:.4f}")
print(f"  Best single fusion ({best_single_name}): F1 = {best_single:.4f}")
print(f"  Δ ensemble − best single: {f1_weighted - best_single:+.4f}")

# %% [markdown]
# ## 3 · Summary CSV

# %%
table_path = OUT / "ensemble_summary.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ensemble", "K", "members", "macro_f1", "delta_vs_best_single"])
    for r in ens_results:
        w.writerow([f"top-{r['K']} fusions" if "label" not in r else r["label"],
                    r["K"], " + ".join(r["members"]),
                    f"{r['f1']:.4f}", f"{r['f1'] - best_single:+.4f}"])
    w.writerow(["weighted F1² (7 fusions)", 7,
                " + ".join(FUSION_DISPLAYS),
                f"{f1_weighted:.4f}", f"{f1_weighted - best_single:+.4f}"])
print(f"\nWrote {table_path}")
