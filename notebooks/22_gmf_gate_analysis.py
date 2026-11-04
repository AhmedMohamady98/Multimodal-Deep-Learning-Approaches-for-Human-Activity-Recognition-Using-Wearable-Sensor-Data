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
# # GMF Gate Activations
#
# The three GMF gates are read directly off the trained checkpoints by
# `jobs/analyze_gmf_gates.py`, which hooks the gate `nn.Linear` layers and
# records the mean sigmoid activation per modality for every test window
# (43,760 windows across the 3 folds). The same run reproduces the reported
# macro F1 of 0.827, so the gates below belong to the model that produced
# the headline result.
#
# Two properties of the architecture shape how these numbers must be read:
#
# 1. The fused vector passes through `LayerNorm` before the classifier, so a
#    uniform rescaling of all three gates leaves the logits unchanged. Only
#    the *ratio* between gates is visible to the model, which is why the
#    per-activity figure below plots each gate's **share** of total gate mass
#    rather than its raw value.
# 2. The gate is a sigmoid with no sparsity penalty in the loss, so it has no
#    reachable zero. "Closed" here means small relative to the others, never
#    literally off.
#
# Outputs → `results/gates/`.

# %%
from notebooks._common import *
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

OUT = FIGURES_ROOT / "gates"
OUT.mkdir(parents=True, exist_ok=True)

GATES = ["g_imu", "g_audio", "g_hum"]
GATE_LABEL = {"g_imu": "IMU", "g_audio": "Audio", "g_hum": "Humidity"}
WRONG_COLOR = "#A9A9A9"

gates = pd.read_csv(OUT / "gmf_gate_activations.csv")
gates["activity"] = gates["true_label"].map(paper_label_for_id)
assert gates["activity"].notna().all(), "unmapped true_label — check paper_names"
gates["correct"] = gates["true_label"] == gates["pred_label"]

# LayerNorm-invariant view: each gate as a fraction of total gate mass.
_total = gates[GATES].sum(axis=1)
for g in GATES:
    gates[f"share_{g}"] = gates[g] / _total

# Attach participant_id and the two-modality predictions from the CSVs behind
# the published results. Both were produced by separate jobs, so requiring the
# true_label sequences to match is a real check that all three sources describe
# the same windows in the same order, not a formality.
_cols = ["participant_id", "fold", "model", "true_label", "pred_label"]
_p3 = pd.read_csv(RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv.gz",
                  usecols=_cols)
_p3 = _p3[_p3["model"] == "gmf"].reset_index(drop=True)
_p2 = pd.read_csv(RESULTS_DIR / "predictions" / "2mod_all_models_raw_predictions.csv.gz",
                  usecols=_cols)
_p2 = _p2[_p2["model"] == "gmf_2mod"].reset_index(drop=True)
for name, frame in [("3-mod", _p3), ("2-mod", _p2)]:
    assert len(frame) == len(gates), f"{name} row count differs from the gate CSV"
    assert (frame["true_label"].values == gates["true_label"].values).all(), \
        f"{name} predictions are not aligned window-for-window with the gate CSV"

gates["participant_id"] = _p3["participant_id"].values
# 'helped' = keeping humidity fixed this window; 'hurt' = keeping it broke one.
gates["correct_2mod"] = gates["true_label"].values == _p2["pred_label"].values
gates["helped"] = gates["correct"] & ~gates["correct_2mod"]
gates["hurt"] = ~gates["correct"] & gates["correct_2mod"]

print(f"Loaded {len(gates):,} windows · {gates['fold'].nunique()} folds · "
      f"{gates['participant_id'].nunique()} participants")
print(gates[GATES].describe().T[["mean", "std", "min", "max"]].to_string())


# %% [markdown]
# ### A note on the unit of analysis
#
# Windows are 500 samples taken at a step of 250, so consecutive windows share
# half their raw signal, and every window from one participant shares a person.
# A test that treats all 43,760 as independent draws will return p-values that
# are far too small to mean anything. Every significance test below therefore
# uses the **participant** as the unit: the statistic is computed within each of
# the 20 participants and compared with a paired signed-rank test across them.

# %%
def paired_over_participants(frame, col, group_a, group_b):
    """Compare `col` between two window subsets, paired within participant.

    Returns (mean difference, n participants, n where A > B, p). The signed-rank
    test over 20 participants bottoms out near 2e-6, so a floor there means
    'consistent in every participant', not 'astronomically significant'.
    """
    per = []
    for _pid, s in frame.groupby("participant_id"):
        a, b = s[group_a(s)], s[group_b(s)]
        if len(a) >= 5 and len(b) >= 5:
            per.append(a[col].mean() - b[col].mean())
    per = np.array(per)
    if len(per) < 5:
        return np.nan, len(per), 0, np.nan
    return per.mean(), len(per), int((per > 0).sum()), wilcoxon(per).pvalue


def fmt_p(p):
    """Format a p-value for figure annotation."""
    if np.isnan(p):
        return "p n/a"
    if p < 1e-4:
        return f"p = {p:.0e}"
    if p < 1e-3:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


# %% [markdown]
# ## 1. Gate distributions across all windows
#
# The central claim this figure supports: the gates *modulate* the three
# streams, they never switch one off. Audio sits high and narrow, IMU in the
# middle, humidity low and by far the widest. The dashed line marks the
# smallest humidity activation observed anywhere in the test set.

# %%
fig, (ax, ax_bar) = plt.subplots(
    2, 1, figsize=(6.4, 4.4), height_ratios=[3.4, 1.0],
    gridspec_kw={"hspace": 0.45},
)

parts = ax.violinplot(
    [gates[g].values for g in GATES],
    positions=[0, 1, 2], vert=False, widths=0.82,
    showextrema=False, showmedians=True,
)
for body, g in zip(parts["bodies"], GATES):
    body.set_facecolor(MODALITY_COLORS[GATE_LABEL[g]])
    body.set_edgecolor("#1a1a1a")
    body.set_linewidth(0.9)
    body.set_alpha(0.85)
parts["cmedians"].set_color("#1a1a1a")
parts["cmedians"].set_linewidth(1.4)

hum_min = gates["g_hum"].min()
ax.axvline(hum_min, color="#1a1a1a", ls="--", lw=0.9, alpha=0.7, zorder=0)
ax.annotate(
    f"smallest humidity gate\nanywhere in the test set: {hum_min:.3f}",
    xy=(hum_min, 2.46), xytext=(hum_min + 0.055, 2.62),
    fontsize=8, ha="left", va="center", color="#1a1a1a",
    arrowprops=dict(arrowstyle="-", lw=0.8, color="#1a1a1a"),
)

for i, g in enumerate(GATES):
    ax.text(gates[g].max() + 0.018, i, f"median {gates[g].median():.3f}",
            va="center", ha="left", fontsize=8, color="#444444")

ax.set_yticks([0, 1, 2])
ax.set_yticklabels([GATE_LABEL[g] for g in GATES])
ax.set_xlabel("Gate activation (mean over the 256 hidden dimensions)")
ax.set_xlim(0, 1)
ax.set_ylim(-0.6, 2.95)
ax.set_title("GMF gate activations over all 43,760 test windows")

# Companion bar: which gate is largest, per window.
argmax = np.asarray(gates[GATES].values).argmax(axis=1)
left = 0.0
for i, g in enumerate(GATES):
    pct = 100 * (argmax == i).mean()
    ax_bar.barh(0, pct, left=left, height=0.55,
                color=MODALITY_COLORS[GATE_LABEL[g]],
                edgecolor="#1a1a1a", linewidth=0.9)
    if pct > 4:
        ax_bar.text(left + pct / 2, 0, f"{GATE_LABEL[g]}  {pct:.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="#1a1a1a", fontweight="semibold")
    else:
        ax_bar.annotate(f"{GATE_LABEL[g]}  {pct:.1f}%",
                        xy=(left + pct / 2, -0.30), xytext=(left + pct / 2, -0.95),
                        ha="center", va="top", fontsize=8, color="#444444",
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#888888"))
    left += pct

ax_bar.set_xlim(0, 100)
ax_bar.set_ylim(-1.5, 0.6)
ax_bar.set_yticks([])
ax_bar.set_xlabel("Share of windows in which this gate is the largest of the three (%)")
ax_bar.spines["left"].set_visible(False)

save_fig(fig, OUT / "gate_distributions")
print(f"argmax shares: " + ", ".join(
    f"{GATE_LABEL[g]} {100 * (argmax == i).mean():.1f}%" for i, g in enumerate(GATES)))


# %% [markdown]
# ## 2. Gate openness against correctness
#
# Windows the model classifies correctly carry a more open gate on **all
# three** modalities, humidity most strongly. Reporting all three is the
# point: the effect is not humidity-specific, so it cannot be read as
# evidence that the humidity gate alone selects informative windows.
#
# The annotation reports the difference in means and how many of the 20
# participants show it in the same direction, rather than a window-level
# p-value.

# %%
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)

for ax, g in zip(axes, GATES):
    ok = gates.loc[gates["correct"], g].values
    bad = gates.loc[~gates["correct"], g].values
    diff, n_par, n_pos, p = paired_over_participants(
        gates, g, lambda s: s["correct"], lambda s: ~s["correct"])
    bins = np.linspace(0, max(gates[g].max(), 0.8), 46)

    ax.hist(bad, bins=bins, density=True, color=WRONG_COLOR,
            alpha=0.85, label="misclassified")
    ax.hist(ok, bins=bins, density=True, histtype="step",
            color=MODALITY_COLORS[GATE_LABEL[g]], lw=1.8, label="correct")
    ax.axvline(bad.mean(), color=WRONG_COLOR, ls="--", lw=1.2)
    ax.axvline(ok.mean(), color=MODALITY_COLORS[GATE_LABEL[g]], ls="--", lw=1.2)

    ax.set_title(f"{GATE_LABEL[g]} gate")
    ax.set_xlabel("Gate activation")
    ax.text(0.97, 0.95,
            f"$\\Delta$ = {diff:+.3f}\n{n_pos}/{n_par} participants",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#222222")

axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("Gate openness on correctly versus incorrectly classified windows",
             y=1.03, fontsize=10.5, fontweight="semibold")

save_fig(fig, OUT / "gate_correct_vs_incorrect")

for g in GATES:
    ok = gates.loc[gates["correct"], g]
    bad = gates.loc[~gates["correct"], g]
    diff, n_par, n_pos, p = paired_over_participants(
        gates, g, lambda s: s["correct"], lambda s: ~s["correct"])
    print(f"{GATE_LABEL[g]:9} correct {ok.mean():.3f}  wrong {bad.mean():.3f}  "
          f"| paired over participants: {diff:+.4f}, "
          f"{n_pos}/{n_par} positive, {fmt_p(p)}")


# %% [markdown]
# ## 3. Humidity gate share by activity
#
# Plotted as a share of total gate mass, because `LayerNorm` makes the
# absolute level invisible to the model. Sorted by median. The width of each
# violin is the substantive point: activity label explains only a third of
# the variance in the humidity gate, so most of the gating happens window to
# window *within* an activity, not between activities.

# %% [markdown]
# Activities are split by whether they involve direct contact with liquid
# water. The assignment is made from the activity alone, and the borderline
# cases are excluded from the test rather than argued over: window cleaning
# and cleaning table may use a damp cloth or spray, and disinfecting hands and
# applying hand cream involve a volatile liquid rather than water.

# %%
WET = ["Washing hands", "Washing dishes", "Watering plants",
       "Brushing Teeth", "Making tea"]
DRY = ["Vacuum cleaning", "Floor cleaning", "Putting away dishes",
       "Cutting vegetables", "Drinking", "Null"]

order = (gates.groupby("activity")["share_g_hum"].median()
         .sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(6.8, 5.8))
parts = ax.violinplot(
    [gates.loc[gates["activity"] == a, "share_g_hum"].values for a in order],
    positions=range(len(order)), vert=False, widths=0.74,
    showextrema=False, showmedians=True,
)
for body, a in zip(parts["bodies"], order):
    body.set_facecolor(MODALITY_COLORS["Humidity"] if a in WET
                       else ("#BFBFBF" if a in DRY else "#E4E4E4"))
    body.set_edgecolor("#1a1a1a")
    body.set_linewidth(0.8)
    body.set_alpha(0.9)
parts["cmedians"].set_color("#1a1a1a")
parts["cmedians"].set_linewidth(1.3)

grand = gates["share_g_hum"].median()
ax.axvline(grand, color="#1a1a1a", ls="--", lw=1.0, alpha=0.75, zorder=0)
ax.text(grand, len(order) - 0.4, f"  overall median {grand:.3f}",
        fontsize=8, color="#444444", va="center", ha="left")

# The claim the colours encode, tested with the participant as the unit.
_diffs = []
for _pid, s in gates.groupby("participant_id"):
    _diffs.append(s.loc[s["activity"].isin(WET), "share_g_hum"].mean()
                  - s.loc[s["activity"].isin(DRY), "share_g_hum"].mean())
_diffs = np.array(_diffs)
_p = wilcoxon(_diffs).pvalue
print(f"wet minus dry, humidity gate share: {_diffs.mean():+.4f}, "
      f"{int((_diffs > 0).sum())}/{len(_diffs)} participants, {fmt_p(_p)}")

import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(facecolor=MODALITY_COLORS["Humidity"], edgecolor="#1a1a1a",
                   label="involves liquid water"),
    mpatches.Patch(facecolor="#BFBFBF", edgecolor="#1a1a1a", label="does not"),
    mpatches.Patch(facecolor="#E4E4E4", edgecolor="#1a1a1a",
                   label="borderline, excluded from the test"),
], loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
    frameon=False, fontsize=8, handlelength=1.3, columnspacing=1.4)

ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("Humidity gate as a share of total gate mass")
ax.set_title("The humidity gate opens widest on activities involving water",
             pad=26)
ax.text(0.015, 0.015,
        f"wet $-$ dry = {_diffs.mean():+.3f},  "
        f"{int((_diffs > 0).sum())}/{len(_diffs)} participants,  {fmt_p(_p)}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        color="#222222")

save_fig(fig, OUT / "humidity_gate_by_activity")

summary = (gates.groupby("activity")
           .agg(n=("share_g_hum", "size"),
                hum_share_median=("share_g_hum", "median"),
                hum_raw_mean=("g_hum", "mean"),
                hum_raw_min=("g_hum", "min"),
                hum_raw_max=("g_hum", "max"))
           .loc[order])
summary.to_csv(OUT / "humidity_gate_by_activity.csv")
print(summary.to_string(float_format="%.3f"))


# %% [markdown]
# ## 4. Reproducibility across folds
#
# The three folds are independently trained models. The correct-versus-wrong
# gap reappears in all three, which rules out a single-fold artefact.

# %%
folds = sorted(gates["fold"].unique())
fig, ax = plt.subplots(figsize=(5.4, 3.0))

width = 0.26
for i, g in enumerate(GATES):
    diffs = []
    for f in folds:
        s = gates[gates["fold"] == f]
        diffs.append(s.loc[s["correct"], g].mean() - s.loc[~s["correct"], g].mean())
    xs = np.arange(len(folds)) + (i - 1) * width
    ax.bar(xs, diffs, width=width, color=MODALITY_COLORS[GATE_LABEL[g]],
           edgecolor="#1a1a1a", linewidth=0.8, label=GATE_LABEL[g])
    for x, d in zip(xs, diffs):
        ax.text(x, d + 0.0015, f"{d:+.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#333333")

ax.set_xticks(range(len(folds)))
ax.set_xticklabels([f"Fold {f}" for f in folds])
ax.set_ylabel("Mean gate, correct $-$ misclassified")
ax.set_ylim(0, max(gates.groupby("fold").apply(
    lambda s: max(s.loc[s["correct"], g].mean() - s.loc[~s["correct"], g].mean()
                  for g in GATES), include_groups=False)) * 1.45)
ax.legend(frameon=False, ncol=3, loc="upper center")
ax.set_title("Correct-versus-misclassified gate gap, per fold", pad=10)

save_fig(fig, OUT / "gate_gap_per_fold")
print("Figures written to", OUT)


# %% [markdown]
# ## 5. Does the gate predict where humidity actually helps?
#
# This is the test the interpretability claim rests on. Every window carries
# both the three-modality and the two-modality prediction, so each one can be
# labelled by what keeping humidity did to it:
#
# - **helped** — three-modality correct, two-modality wrong
# - **hurt** — three-modality wrong, two-modality correct
#
# If the gate admits humidity where humidity is informative, `g_hum` should be
# higher on helped windows than on hurt ones, and classes that benefit should
# be classes with an open humidity gate. One caveat applies throughout: the
# two-modality model is separately trained, so this contrast contains ordinary
# run-to-run variation as well as any effect of the humidity stream.

# %%
n_helped, n_hurt = int(gates["helped"].sum()), int(gates["hurt"].sum())
print(f"helped {n_helped:,} · hurt {n_hurt:,} · "
      f"net {n_helped - n_hurt:+,} of {len(gates):,} windows "
      f"({100 * (n_helped - n_hurt) / len(gates):+.2f}%)")

for g in GATES:
    diff, n_par, n_pos, p = paired_over_participants(
        gates, g, lambda s: s["helped"], lambda s: s["hurt"])
    print(f"  {GATE_LABEL[g]:9} helped-minus-hurt {diff:+.4f}   "
          f"{n_pos}/{n_par} participants positive   {fmt_p(p)}")

per_class = pd.DataFrame([
    {"activity": paper_label_for_id(k),
     "n": len(s),
     "mean g_hum": s["g_hum"].mean(),
     "mean humidity share": s["share_g_hum"].mean(),
     "net rate %": 100 * (s["helped"].sum() - s["hurt"].sum()) / len(s)}
    for k, s in gates.groupby("true_label")
])
rho, rho_p = spearmanr(per_class["mean humidity share"], per_class["net rate %"])
print(f"\nSpearman(humidity gate share, net benefit) = {rho:+.3f}  p = {rho_p:.3f}"
      f"   (n = {len(per_class)} classes)")

# %%
fig, ax = plt.subplots(figsize=(7.0, 5.0))

ax.axhline(0, color="#888888", lw=0.9, ls="-", zorder=0)
ax.scatter(per_class["mean humidity share"], per_class["net rate %"],
           s=np.sqrt(per_class["n"]) * 3.2,
           color=MODALITY_COLORS["Humidity"], edgecolor="#1a1a1a",
           linewidth=0.9, alpha=0.9, zorder=3)
ax.margins(x=0.16, y=0.13)

# Half the classes sit in a tight band around share 0.27, so labels need
# placing rather than offsetting uniformly. For each point, score the four
# candidate sides by how many other points fall in that direction (in
# axis-normalised coordinates, since the two scales differ by two orders of
# magnitude) and take the emptiest.
_x = per_class["mean humidity share"].to_numpy()
_y = per_class["net rate %"].to_numpy()
_nx = (_x - _x.min()) / np.ptp(_x)
_ny = (_y - _y.min()) / np.ptp(_y)

CANDIDATES = {                      # side: (offset points, ha, va)
    "above": ((0, 10), "center", "bottom"),
    "below": ((0, -11), "center", "top"),
    "right": ((11, -3), "left", "center"),
    "left":  ((-11, -3), "right", "center"),
}

for i, name in enumerate(per_class["activity"]):
    dx, dy = _nx - _nx[i], _ny - _ny[i]
    near = np.sqrt(dx ** 2 + dy ** 2) < 0.22
    near[i] = False
    crowd = {
        "above": int((near & (dy > 0) & (abs(dx) < 0.16)).sum()),
        "below": int((near & (dy < 0) & (abs(dx) < 0.16)).sum()),
        "right": int((near & (dx > 0)).sum()) + 3 * (_nx[i] > 0.86),
        "left":  int((near & (dx < 0)).sum()) + 3 * (_nx[i] < 0.14),
    }
    side = min(crowd, key=crowd.get)
    offset, ha, va = CANDIDATES[side]
    ax.annotate(name, (_x[i], _y[i]), textcoords="offset points",
                xytext=offset, ha=ha, va=va, fontsize=7.2, color="#333333")

ax.set_xlabel("Mean humidity gate share for the activity")
ax.set_ylabel("Net windows humidity fixed, as % of the activity")
ax.set_title("The humidity gate does not track where humidity helps")
ax.text(0.985, 0.975,
        f"Spearman $\\rho$ = {rho:+.3f},  {fmt_p(rho_p)},  n = {len(per_class)}"
        "\nmarker area $\\propto$ number of windows",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        color="#222222", linespacing=1.5)

save_fig(fig, OUT / "gate_vs_humidity_benefit")
per_class.sort_values("net rate %", ascending=False).to_csv(
    OUT / "gate_vs_humidity_benefit.csv", index=False)
print(per_class.sort_values("net rate %", ascending=False)
      .to_string(index=False, float_format="%.4f"))


# %% [markdown]
# ## 6. Workbook export
#
# Everything above, plus two diagnostic sheets, written to a single `.xlsx`:
#
# - **Sigmoid floor** — the pre-sigmoid value the gate layer would have to emit
#   to reach a given activation, and the sigmoid gradient there. Explains why
#   no gate reaches zero: the gradient that would drive it there vanishes as it
#   approaches, and the loss carries no sparsity term to supply the pressure.
# - **Variance decomposition** — how much of each gate's variance is explained
#   by the activity label versus varying window to window within an activity.

# %%
def variance_split(col):
    """Share of a gate's total sum of squares explained by the activity label."""
    grand = gates[col].mean()
    grp = gates.groupby("true_label")[col]
    between = (grp.count() * (grp.mean() - grand) ** 2).sum()
    total = ((gates[col] - grand) ** 2).sum()
    return 100 * between / total


argmax = np.asarray(gates[GATES].values).argmax(axis=1)
sheets = {}

sheets["Per-class gate means"] = pd.DataFrame([
    {"activity": a, "n_windows": len(s),
     **{f"{GATE_LABEL[g]} mean": s[g].mean() for g in GATES},
     **{f"{GATE_LABEL[g]} std": s[g].std() for g in GATES},
     **{f"{GATE_LABEL[g]} share": s[f"share_{g}"].mean() for g in GATES}}
    for a, s in gates.groupby("activity")
]).sort_values("Humidity mean", ascending=False)

rows = []
for label, s in [("ALL CLASSES POOLED", gates), *gates.groupby("activity")]:
    ok, bad = s[s["correct"]], s[~s["correct"]]
    if len(ok) < 5 or len(bad) < 5:
        continue
    r = {"activity": label, "n_correct": len(ok), "n_wrong": len(bad)}
    for g in GATES:
        diff, n_par, n_pos, p = paired_over_participants(
            s, g, lambda x: x["correct"], lambda x: ~x["correct"])
        r[f"{GATE_LABEL[g]} correct"] = ok[g].mean()
        r[f"{GATE_LABEL[g]} wrong"] = bad[g].mean()
        r[f"{GATE_LABEL[g]} diff"] = ok[g].mean() - bad[g].mean()
        r[f"{GATE_LABEL[g]} participants +/n"] = f"{n_pos}/{n_par}"
        r[f"{GATE_LABEL[g]} p (paired, participant)"] = p
    rows.append(r)
sheets["Correct vs incorrect"] = pd.DataFrame(rows)

rows = []
for g in GATES:
    diff, n_par, n_pos, p = paired_over_participants(
        gates, g, lambda s: s["helped"], lambda s: s["hurt"])
    rows.append({
        "gate": GATE_LABEL[g],
        "mean on helped windows": gates.loc[gates["helped"], g].mean(),
        "mean on hurt windows": gates.loc[gates["hurt"], g].mean(),
        "helped minus hurt (paired)": diff,
        "participants positive": f"{n_pos}/{n_par}",
        "p (paired, participant)": p,
    })
sheets["Humidity benefit test"] = pd.concat(
    [pd.DataFrame(rows),
     pd.DataFrame([{"gate": "--- per class ---"}]),
     per_class.rename(columns={"activity": "gate"})],
    ignore_index=True)

sheets["Gate distribution"] = pd.DataFrame([
    {"gate": GATE_LABEL[g], "mean": gates[g].mean(), "std": gates[g].std(),
     "min": gates[g].min(), "median": gates[g].median(), "max": gates[g].max(),
     "% windows this gate is largest": 100 * (argmax == i).mean(),
     **{f"% below {t}": 100 * (gates[g] < t).mean()
        for t in (0.30, 0.20, 0.10, 0.05, 0.01)}}
    for i, g in enumerate(GATES)
])

sheets["Sigmoid floor"] = pd.DataFrame([
    {"target gate value": t,
     "required pre-sigmoid value": np.log(t / (1 - t)),
     "sigmoid gradient there": t * (1 - t),
     "gradient vs its peak at 0.5": t * (1 - t) / 0.25}
    for t in (0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001)
])

sheets["Variance decomposition"] = pd.DataFrame([
    {"gate": GATE_LABEL[g],
     "between-activity share %": variance_split(g),
     "within-activity share %": 100 - variance_split(g),
     "range across activity means": (gates.groupby("true_label")[g].mean().max()
                                     - gates.groupby("true_label")[g].mean().min()),
     "range across windows": gates[g].max() - gates[g].min()}
    for g in GATES
])

sheets["Per-fold"] = pd.DataFrame([
    {"fold": int(f), "n_windows": len(s),
     **{f"{GATE_LABEL[g]} mean": s[g].mean() for g in GATES},
     **{f"{GATE_LABEL[g]} correct-wrong diff":
        s.loc[s["correct"], g].mean() - s.loc[~s["correct"], g].mean()
        for g in GATES}}
    for f, s in gates.groupby("fold")
])

raw = gates[["fold", "activity", "true_label", "pred_label", "correct"] + GATES].copy()
raw["predicted_activity"] = raw["pred_label"].map(paper_label_for_id)
sheets["Raw windows"] = raw

xlsx = OUT / "gmf_gate_activations.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    for name, frame in sheets.items():
        frame.to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]
        ws.freeze_panes = "A2"
        for column in ws.columns:
            widest = max((len(str(c.value)) for c in column[:200]
                          if c.value is not None), default=0)
            ws.column_dimensions[column[0].column_letter].width = min(
                max(widest + 2, 10), 32)

print(f"Wrote {xlsx}")
for name, frame in sheets.items():
    print(f"  {name:24} {len(frame):>6,} rows")
print("\n" + sheets["Variance decomposition"].to_string(index=False,
                                                        float_format="%.1f"))
