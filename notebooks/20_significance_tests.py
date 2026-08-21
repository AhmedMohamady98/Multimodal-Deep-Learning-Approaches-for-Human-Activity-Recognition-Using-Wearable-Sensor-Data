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
# # Paired Significance Tests — is the fusion ranking statistically supported?
#
# The headline claim of the comparison is an *ordering* of fusion methods. An
# ordering read off mean macro F1 alone says nothing about whether neighbouring
# methods are actually separable. This notebook supplies the missing inference.
#
# **Unit of analysis.** Under 3-fold group cross-validation every participant is
# held out in exactly one fold, so each model produces exactly one macro F1 per
# participant, computed on an identical window set. The 20 per-participant scores
# therefore form a genuine paired sample across models.
#
# **Procedure.** For each comparison we test the 20 paired differences. A
# Shapiro-Wilk test checks the normality assumption of the paired t-test; the
# Wilcoxon signed-rank test is reported alongside as the distribution-free
# equivalent, and an exact sign test as the weakest-assumption fallback. Effect
# sizes accompany every p-value, and families of comparisons are Holm-corrected.
#
# Source: `predictions/all_models_raw_predictions.csv`.
# Outputs → `results/v3/10s/significance/`.
#
# > **Tweaking the figures.** Every colour, font size, label and dimension lives
# > in the single config cell in section 0. Edit it, re-run that cell, then re-run
# > the two figure cells (sections 7 and 8). Nothing else needs recomputing, and
# > the statistics are untouched by anything in the config.

# %%
# Put the repo root on sys.path so `notebooks.v3._common` resolves regardless of
# where the kernel starts. Jupyter starts it in this file's directory; running
# the paired .py from the repo root starts it there. Both work.
import sys
from pathlib import Path

_cwd = Path.cwd().resolve()
for _cand in (_cwd, *_cwd.parents):
    if (_cand / "notebooks" / "v3" / "_common.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from notebooks.v3._common import *
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

OUT = FIGURES_ROOT / "significance"
OUT.mkdir(parents=True, exist_ok=True)
print(f"Repo root: {_cand}\nOutputs  : {OUT}")

# %% [markdown]
# ## 0 · Configuration — everything adjustable lives here
#
# Edit any value below and re-run this cell, then re-run the figure cells.
# The statistics in sections 1 to 5 do not depend on any of it.

# %%
# ── Statistics ───────────────────────────────────────────────────────────────
ALPHA     = 0.05          # significance threshold
N_BOOT    = 10_000        # bootstrap resamples for the CI on the mean difference
BOOT_SEED = 20260805      # fixed so the reported CI is reproducible

# Equivalence bounds for the TOST in section 3.1, in macro F1 points. A bound is
# the largest difference you would still call "practically the same".
EQUIV_BOUNDS  = (0.010, 0.020, 0.030, 0.050)
POWER_TARGETS = (0.80, 0.90)   # power levels to report required sample sizes for

# ── Output ───────────────────────────────────────────────────────────────────
SAVE_FORMATS = ("svg", "pdf")   # add "png" for a quick raster preview
SAVE_CLOSE   = False            # False keeps figures visible inline in the notebook

# ── Font sizes (applied as rcParams, so they affect both figures) ────────────
FONT = {
    "title":       10.5,   # axes titles
    "axis_label":   9.5,   # x/y axis labels
    "tick":         8.5,   # tick labels
    "legend":       9.0,   # legend entries
    "annot":        8.0,   # numbers written inside the heatmap
    "participant":  8.0,   # the P01…P20 tick labels
}

# ── Colours ──────────────────────────────────────────────────────────────────
# FUSION_COLORS comes from src/viz/paper_style.py and keeps these consistent
# with every other figure in the paper. Override here to depart from it.
COLORS = {
    "gmf":         FUSION_COLORS.get("GMF", "#2A9D8F"),
    "late":        FUSION_COLORS.get("Late Fusion", "#264653"),
    "connector":   "#BBBBBB",   # the line joining each participant's two dots
    "left_hander": "#E76F51",   # highlight for P07 / P10 / P14
    "mean_line":   "#E76F51",   # dashed mean-difference marker
    "zero_line":   "#444444",   # the x = 0 reference
    "marker_edge": "white",
}

# ── Text ─────────────────────────────────────────────────────────────────────
LABELS = {
    "gmf":            "GMF",
    "late":           "Late Fusion",
    "f1_axis":        "Macro F1",
    "diff_axis":      "GMF − Late Fusion (macro F1)",
    "paired_title":   "Per-participant scores (left-handers in terracotta)",
    "cbar":           "Holm-adjusted p",
    "matrix_title":   ("Pairwise separability — Wilcoxon signed-rank over "
                       "20 participants\n(green = separable at 0.05, "
                       "red = indistinguishable)"),
    "participant_fmt": "P{:02d}",     # how participant ticks are written
}

# ── Figure geometry ──────────────────────────────────────────────────────────
FIG = {
    "paired_size":       (12.5, 5.4),   # width, height in inches
    "paired_ratios":     [1.65, 1],     # left panel : right panel width
    "matrix_size":       (7.6, 6.2),
    "dot_size":          42,            # dumbbell marker area
    "connector_width":   1.4,
    "xtick_rotation":    28,            # rotation of model names on the matrix
}

# ── Heatmap ──────────────────────────────────────────────────────────────────
HEATMAP = {
    "cmap":      "RdYlGn_r",   # try "RdBu_r", "coolwarm", or CM_CMAP
    "vmin":      0.0,
    "vmax":      0.2,          # p-values above this all read as fully "red"
    "fmt":       ".3f",        # annotation number format
    "linewidth": 0.5,
    "linecolor": "white",
}

# ── Model ordering and display names ─────────────────────────────────────────
# Rename freely, e.g. "Late Fusion" -> "Late Concatenation". Keys must stay as
# they are, since they index the results; only the values are shown.
FUSION_ORDER = ["GMF", "Late Fusion", "CMA", "CLS Transformer",
                "MBT", "LMF", "Decision Fusion"]
DISPLAY_NAMES = {m: m for m in FUSION_ORDER}

# Apply the font settings globally.
plt.rcParams.update({
    "axes.titlesize":  FONT["title"],
    "axes.labelsize":  FONT["axis_label"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],
})
print("Config applied. Re-run the figure cells (sections 7 and 8) to see changes.")

# %% [markdown]
# ## 1 · Load and verify the per-participant scores
#
# Recomputed from the raw predictions rather than read off a summary table, so
# the tests run on values traceable to the model outputs.

# %%
preds = pd.read_csv(
    RESULTS_DIR / "predictions" / "all_models_raw_predictions.csv",
    usecols=["participant_id", "fold", "model", "true_label", "pred_label"],
)


def macro_f1(df):
    if len(df) == 0:
        return float("nan")
    return f1_score(df["true_label"], df["pred_label"], average="macro",
                    labels=list(range(NUM_CLASSES)), zero_division=0)


rows = []
for p in sorted(preds["participant_id"].unique()):
    for display, key in MODEL_CSV_KEYS.items():
        sub = preds[(preds["model"] == key) & (preds["participant_id"] == p)]
        rows.append({"participant": p, "model": display, "f1": macro_f1(sub),
                     "fold": int(sub["fold"].iloc[0])})
long = pd.DataFrame(rows)
mat = (long.pivot(index="participant", columns="model", values="f1")
           .reindex(columns=ALL_MODELS_ORDER))
fold_of = long.groupby("participant")["fold"].first()
print(f"Matrix: {mat.shape[0]} participants × {mat.shape[1]} models")
mat.round(4)

# %% [markdown]
# ## 2 · The paired test, in full
#
# `paired_report` runs the whole battery on one pair of models so that every
# comparison in this notebook is produced identically.

# %%
def paired_report(a, b, name_a, name_b):
    """Complete paired comparison of two models over participants."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)

    sw_W, sw_p = stats.shapiro(d)
    t_stat, t_p = stats.ttest_rel(a, b)
    w_stat, w_p = stats.wilcoxon(a, b, method="exact")

    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - ALPHA / 2, n - 1)

    ranks = stats.rankdata(np.abs(d))
    rank_pos, rank_neg = ranks[d > 0].sum(), ranks[d < 0].sum()

    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())

    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)

    return {
        "a": name_a, "b": name_b, "n": n,
        "mean_a": a.mean(), "mean_b": b.mean(),
        "mean_diff": d.mean(), "median_diff": float(np.median(d)), "sd_diff": sd,
        "n_pos": n_pos, "n_neg": n_neg, "n_tie": int((d == 0).sum()),
        "shapiro_W": sw_W, "shapiro_p": sw_p, "normal": bool(sw_p > ALPHA),
        "t": t_stat, "t_p": t_p, "cohen_dz": d.mean() / sd,
        "ci_lo": d.mean() - tcrit * se, "ci_hi": d.mean() + tcrit * se,
        "boot_lo": float(np.percentile(boot, 2.5)),
        "boot_hi": float(np.percentile(boot, 97.5)),
        "W": w_stat, "wilcoxon_p": w_p,
        "rank_biserial": (rank_pos - rank_neg) / ranks.sum(),
        "sign_p": stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue,
    }


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


# %% [markdown]
# ## 3 · Primary comparison — GMF vs Late Fusion
#
# The one comparison the ranking actually hinges on. Everything below it in the
# table is separated by a comfortable margin; the top two are not.

# %%
gmf, late = mat["GMF"].to_numpy(), mat["Late Fusion"].to_numpy()
primary = paired_report(gmf, late, "GMF", "Late Fusion")

per_p = pd.DataFrame({
    "fold": fold_of.to_numpy(),
    "GMF": gmf,
    "Late Fusion": late,
    "diff": gmf - late,
}, index=mat.index)
per_p["winner"] = np.where(per_p["diff"] > 0, "GMF", "Late Fusion")
per_p["left_handed"] = [p in LEFT_HANDERS for p in per_p.index]
print(per_p.round(4).to_string())

print(f"""
  GMF mean over participants   {primary['mean_a']:.4f}
  Late Fusion mean             {primary['mean_b']:.4f}
  mean paired difference       {primary['mean_diff']:+.4f}  (sd {primary['sd_diff']:.4f})
  median paired difference     {primary['median_diff']:+.4f}
  GMF wins on {primary['n_pos']}/{primary['n']} participants

  Shapiro-Wilk (differences)   W = {primary['shapiro_W']:.4f}, p = {primary['shapiro_p']:.4f}
    -> differences {'are' if primary['normal'] else 'are NOT'} consistent with normality
  Paired t-test                t({primary['n']-1}) = {primary['t']:.3f}, p = {primary['t_p']:.4f}
    Cohen's dz                 {primary['cohen_dz']:.3f}
    95% CI                     [{primary['ci_lo']:+.4f}, {primary['ci_hi']:+.4f}]
    95% CI (bootstrap)         [{primary['boot_lo']:+.4f}, {primary['boot_hi']:+.4f}]
  Wilcoxon signed-rank         W = {primary['W']:.1f}, p = {primary['wilcoxon_p']:.4f} (exact)
    rank-biserial r            {primary['rank_biserial']:.3f}
  Exact sign test              p = {primary['sign_p']:.4f}
""")

# Participants inside a fold share one trained model, so the 20 differences are
# clustered. Reported as a descriptive caveat, not as a separate test.
print("  per-fold breakdown (participants in a fold share a trained model):")
for f in sorted(per_p["fold"].unique()):
    sub = per_p[per_p["fold"] == f]
    print(f"    fold {f}: n={len(sub)}  mean diff {sub['diff'].mean():+.4f}  "
          f"GMF wins {int((sub['diff'] > 0).sum())}/{len(sub)}")

# %% [markdown]
# ## 3.1 · Is that result conclusive? Power and equivalence
#
# A p-value above the threshold is easy to misread as "the two methods are the
# same". It is not. Failing to reject the null says nothing on its own about
# whether an effect exists, so this section asks two further questions.
#
# 1. **What is the smallest difference this design could have detected?** This
#    is a sensitivity analysis. The minimum detectable effect depends only on
#    the sample size, the variability and the chosen power, not on the result
#    we happened to observe, so it is a property of the study rather than a
#    restatement of the p-value.
# 2. **Can equivalence be claimed positively?** That needs a two one-sided test
#    (TOST) against an explicit bound, not the absence of a significant
#    difference.
#
# > **Why observed power is not reported here.** A natural thing to compute is
# > the power to detect the effect we actually measured. We deliberately do not
# > report it. Observed power is a one-to-one function of the p-value, so it
# > carries no information beyond it, and arguing that a null result is
# > explained by low observed power is circular (Hoenig & Heisey, 2001, *The
# > Abuse of Power*). The minimum detectable effect below answers the same
# > question without that objection, and is the statistic to quote.

# %%
def paired_power(effect_dz, n, alpha=ALPHA):
    """Power of a two-sided paired t-test for a given standardised effect."""
    df = n - 1
    crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = effect_dz * np.sqrt(n)
    return (1 - stats.nct.cdf(crit, df, ncp)) + stats.nct.cdf(-crit, df, ncp)


def tost(d, bound, alpha=ALPHA):
    """Two one-sided tests for equivalence within +/- bound. Returns p."""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lower = 1 - stats.t.cdf((d.mean() + bound) / se, n - 1)
    p_upper = stats.t.cdf((d.mean() - bound) / se, n - 1)
    return max(p_lower, p_upper)


d_diff = per_p["diff"].to_numpy()
n_obs = len(d_diff)
dz_obs = primary["cohen_dz"]
sd_obs = primary["sd_diff"]

# Smallest standardised effect this n can detect at 80% power.
mde_dz = next(e for e in np.arange(0.01, 3.0, 0.001)
              if paired_power(e, n_obs) >= 0.80)

print("SENSITIVITY — what this design could detect  [quote these]")
print(f"  observed effect                                    d_z = {dz_obs:.3f} "
      f"({primary['mean_diff']:+.4f} macro F1)")
print(f"  smallest effect detectable at 80% power with n={n_obs}  "
      f"d_z = {mde_dz:.3f}  ({mde_dz * sd_obs:.4f} macro F1)")
if mde_dz * sd_obs > abs(primary["mean_diff"]):
    print(f"  -> the observed lead ({abs(primary['mean_diff']):.4f}) is BELOW the "
          f"detection threshold ({mde_dz * sd_obs:.4f}).")
    print("     A non-significant result here is inconclusive, not evidence of equality.")

# The MDE depends on the sd, which is itself estimated from 20 differences.
# Check the conclusion survives that uncertainty rather than resting on a point
# estimate.
sd_lo = sd_obs * np.sqrt((n_obs - 1) / stats.chi2.ppf(0.975, n_obs - 1))
sd_hi = sd_obs * np.sqrt((n_obs - 1) / stats.chi2.ppf(0.025, n_obs - 1))
print(f"  95% CI on the sd [{sd_lo:.4f}, {sd_hi:.4f}] -> MDE in "
      f"[{mde_dz * sd_lo:.4f}, {mde_dz * sd_hi:.4f}]")
print(f"  -> observed lead stays below the MDE across that whole range: "
      f"{abs(primary['mean_diff']) < mde_dz * sd_lo}")

print("\nSAMPLE SIZE — conditional on the true effect equalling the observed one")
for target in POWER_TARGETS:
    n_req = next(k for k in range(5, 1000) if paired_power(dz_obs, k) >= target)
    print(f"  participants needed for {target:.0%} power if true d_z = {dz_obs:.3f}   {n_req}")


print("\nEQUIVALENCE (TOST)")
for bound in EQUIV_BOUNDS:
    p_t = tost(d_diff, bound)
    print(f"  within +/-{bound:.3f} macro F1   p = {p_t:.4f}   "
          f"{'equivalent' if p_t < ALPHA else 'cannot claim equivalence'}")

print("\nDIRECTION (reported for completeness; a one-sided test chosen after")
print("seeing the data is not defensible, so the two-sided result stands)")
os_t = stats.ttest_rel(per_p["GMF"], per_p["Late Fusion"], alternative="greater")
os_w = stats.wilcoxon(per_p["GMF"], per_p["Late Fusion"],
                      alternative="greater", method="exact")
print(f"  one-sided paired t-test   t = {os_t.statistic:.3f}, p = {os_t.pvalue:.4f}")
print(f"  one-sided Wilcoxon        W = {os_w.statistic:.1f}, p = {os_w.pvalue:.4f}")

power_rows = [{
    "observed_dz": dz_obs,
    "observed_diff": primary["mean_diff"],
    "mde_dz_80pct": mde_dz,
    "mde_f1_80pct": mde_dz * sd_obs,
    **{f"n_for_{int(t*100)}pct_power":
       next(k for k in range(5, 1000) if paired_power(dz_obs, k) >= t)
       for t in POWER_TARGETS},
    **{f"tost_p_{b:.3f}": tost(d_diff, b) for b in EQUIV_BOUNDS},
    "one_sided_t_p": os_t.pvalue,
    "one_sided_wilcoxon_p": os_w.pvalue,
}]
power_df = pd.DataFrame(power_rows)
power_df.round(4)

# %% [markdown]
# ## 4 · GMF against every other fusion method

# %%
vs_gmf = [paired_report(mat["GMF"], mat[m], "GMF", m)
          for m in FUSION_ORDER if m != "GMF"]
gmf_df = pd.DataFrame(vs_gmf)[
    ["b", "mean_diff", "n_pos", "shapiro_p", "t_p", "wilcoxon_p", "cohen_dz"]
].rename(columns={"b": "vs", "n_pos": "GMF wins"})
gmf_df["t_p_holm"] = holm([r["t_p"] for r in vs_gmf])
gmf_df["wilcoxon_p_holm"] = holm([r["wilcoxon_p"] for r in vs_gmf])
gmf_df.round(4)

# %% [markdown]
# ## 5 · Every fusion method against the best unimodal baseline (AST)
#
# The paper states that every fusion method clears AST. Numerically it does.
# Whether the margin survives a paired test is a separate question, and for the
# two weakest methods it does not.

# %%
vs_ast = [paired_report(mat[m], mat["AST"], m, "AST") for m in FUSION_ORDER]
ast_df = pd.DataFrame(vs_ast)[
    ["a", "mean_diff", "n_pos", "wilcoxon_p", "cohen_dz"]
].rename(columns={"a": "method", "n_pos": "wins"})
ast_df["wilcoxon_p_holm"] = holm([r["wilcoxon_p"] for r in vs_ast])
ast_df["separable"] = ast_df["wilcoxon_p_holm"] < ALPHA
ast_df.round(4)

# %% [markdown]
# ## 6 · All 21 pairwise fusion comparisons
#
# The methods group into tiers rather than a strict order: every cross-tier pair
# separates cleanly, and no within-tier pair does.

# %%
pairwise = [paired_report(mat[m1], mat[m2], m1, m2)
            for i, m1 in enumerate(FUSION_ORDER) for m2 in FUSION_ORDER[i + 1:]]
pw_df = pd.DataFrame(pairwise)[
    ["a", "b", "mean_diff", "n_pos", "wilcoxon_p", "cohen_dz"]
].rename(columns={"a": "A", "b": "B", "n_pos": "A wins"})
pw_df["wilcoxon_p_holm"] = holm([r["wilcoxon_p"] for r in pairwise])
pw_df["separable"] = pw_df["wilcoxon_p_holm"] < ALPHA

n_sep = int(pw_df["separable"].sum())
print(f"  {n_sep}/{len(pw_df)} pairs separable at alpha={ALPHA} after Holm correction")
print("  NOT separable: " + ", ".join(
    f"{r.A} vs {r.B}" for r in pw_df[~pw_df["separable"]].itertuples()))
pw_df.sort_values("wilcoxon_p").round(4)

# %% [markdown]
# ## 7 · Figure — the paired difference, participant by participant
#
# A dumbbell of the 20 paired scores plus the distribution of their differences.
# Reading the two panels together shows why the mean gap is not decisive: the
# difference changes sign on 8 of 20 participants.
#
# *Adjust appearance in the config cell (section 0), then re-run this cell.*

# %%
fig, axes = plt.subplots(1, 2, figsize=FIG["paired_size"],
                         gridspec_kw={"width_ratios": FIG["paired_ratios"]})

# ── Left panel: dumbbell of the paired scores ────────────────────────────────
ax = axes[0]
ys = np.arange(len(per_p))
for y, (_, r) in zip(ys, per_p.iterrows()):
    ax.plot([r["Late Fusion"], r["GMF"]], [y, y], color=COLORS["connector"],
            linewidth=FIG["connector_width"], zorder=1, solid_capstyle="round")
ax.scatter(per_p["Late Fusion"], ys, s=FIG["dot_size"], color=COLORS["late"],
           label=LABELS["late"], zorder=3,
           edgecolor=COLORS["marker_edge"], linewidth=0.6)
ax.scatter(per_p["GMF"], ys, s=FIG["dot_size"], color=COLORS["gmf"],
           label=LABELS["gmf"], zorder=3,
           edgecolor=COLORS["marker_edge"], linewidth=0.6)
ax.set_yticks(ys)
ax.set_yticklabels([LABELS["participant_fmt"].format(p) for p in per_p.index],
                   fontsize=FONT["participant"])
for tl, lh in zip(ax.get_yticklabels(), per_p["left_handed"]):
    if lh:
        tl.set_color(COLORS["left_hander"])
        tl.set_weight("semibold")
ax.invert_yaxis()
ax.set_xlabel(LABELS["f1_axis"])
ax.set_title(LABELS["paired_title"])
ax.legend(frameon=False, loc="lower right", fontsize=FONT["legend"])

# ── Right panel: sorted paired differences ───────────────────────────────────
ax = axes[1]
d = per_p["diff"].to_numpy()
order = np.argsort(d)
bar_colors = [COLORS["gmf"] if v > 0 else COLORS["late"] for v in d]
ax.barh(np.arange(len(d)), d[order], color=[bar_colors[i] for i in order],
        edgecolor=COLORS["marker_edge"], linewidth=0.5)
ax.axvline(0, color=COLORS["zero_line"], linewidth=1.0)
ax.axvline(primary["mean_diff"], color=COLORS["mean_line"], linestyle="--",
           linewidth=1.3, label=f"mean {primary['mean_diff']:+.4f}")
ax.set_yticks(np.arange(len(d)))
ax.set_yticklabels([LABELS["participant_fmt"].format(per_p.index[i]) for i in order],
                   fontsize=FONT["participant"])
for tl, i in zip(ax.get_yticklabels(), order):
    if per_p["left_handed"].iloc[i]:
        tl.set_color(COLORS["left_hander"])
        tl.set_weight("semibold")
ax.set_xlabel(LABELS["diff_axis"])
ax.set_title(f"Paired differences: {LABELS['gmf']} wins "
             f"{primary['n_pos']}/{primary['n']}\n"
             f"Wilcoxon p = {primary['wilcoxon_p']:.3f}, "
             f"t-test p = {primary['t_p']:.3f}")
ax.legend(frameon=False, loc="lower right", fontsize=FONT["legend"])

plt.tight_layout()
save_fig(fig, OUT / "gmf_vs_late_paired",
         formats=SAVE_FORMATS, close=SAVE_CLOSE)

# %% [markdown]
# ## 8 · Figure — which pairs are actually separable
#
# Holm-corrected Wilcoxon p-values for all 21 pairs, laid out as a matrix. The
# tiers read off as the coloured blocks against the diagonal.
#
# *Adjust appearance in the config cell (section 0), then re-run this cell.*

# %%
pmat = pd.DataFrame(np.nan, index=FUSION_ORDER, columns=FUSION_ORDER)
for r, p_adj in zip(pairwise, pw_df["wilcoxon_p_holm"]):
    pmat.loc[r["a"], r["b"]] = p_adj
    pmat.loc[r["b"], r["a"]] = p_adj
pmat = pmat.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)

fig, ax = plt.subplots(figsize=FIG["matrix_size"])
sns.heatmap(pmat, ax=ax, cmap=HEATMAP["cmap"],
            vmin=HEATMAP["vmin"], vmax=HEATMAP["vmax"], mask=pmat.isna(),
            annot=True, fmt=HEATMAP["fmt"],
            linewidths=HEATMAP["linewidth"], linecolor=HEATMAP["linecolor"],
            cbar_kws={"label": LABELS["cbar"]},
            annot_kws={"size": FONT["annot"]})
ax.set_xticklabels(ax.get_xticklabels(), rotation=FIG["xtick_rotation"],
                   ha="right", fontsize=FONT["tick"])
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FONT["tick"])
ax.set_title(LABELS["matrix_title"])
plt.tight_layout()
save_fig(fig, OUT / "pairwise_separability",
         formats=SAVE_FORMATS, close=SAVE_CLOSE)

# %% [markdown]
# ## 9 · Tables

# %%
mat.to_csv(OUT / "per_participant_f1.csv", float_format="%.6f")
per_p.to_csv(OUT / "gmf_vs_late_per_participant.csv", float_format="%.6f")
pd.DataFrame([primary]).to_csv(OUT / "primary_test.csv", index=False,
                               float_format="%.6f")
gmf_df.to_csv(OUT / "gmf_vs_all.csv", index=False, float_format="%.6f")
ast_df.to_csv(OUT / "fusion_vs_ast.csv", index=False, float_format="%.6f")
pw_df.to_csv(OUT / "pairwise_all.csv", index=False, float_format="%.6f")
power_df.to_csv(OUT / "power_and_equivalence.csv", index=False, float_format="%.6f")
print(f"Wrote 7 CSVs to {OUT}")
