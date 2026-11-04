"""
Late Fusion 10s evaluation script.

Loads late_fusion_fold{0,1,2}_best.pt, runs inference, generates comprehensive
high-quality figures (300 DPI, seaborn ticks, value labels on every bar) and
tables. Figures compare Late Fusion against unimodal baselines loaded from
results/unimodal_results.json.

Output structure:
    results/figures/late_fusion/
    ├── training_curves.png
    ├── per_fold_f1.png
    ├── per_class_f1.png
    ├── confusion_matrix.png
    ├── fusion_vs_unimodal.png
    ├── per_class_comparison.png
    ├── lift_over_best_unimodal.png
    ├── fold0/ {confusion_matrix, per_class_f1}.png
    ├── fold1/ ...
    └── fold2/ ...

    results/tables/
    ├── late_fusion_summary.csv
    ├── late_fusion_per_fold.csv
    ├── late_fusion_per_class_f1.csv
    └── late_fusion_most_confused.csv

    results/late_fusion_results.json

Usage:
    PYTHONPATH=. python jobs/evaluate_late_fusion_10s.py
"""

import os
import re
import csv
import glob
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.data.splits import get_3fold_splits
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.late_fusion import LateFusion

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR     = "results/models"
FIGURES_DIR    = "results/figures/late_fusion"
TABLES_DIR     = "results/tables"
JSON_OUT       = "results/late_fusion_results.json"
UNIMODAL_JSON  = "results/unimodal_results.json"

WINDOW_SIZE = 500
STEP        = 250
EMBED_DIM   = 128

CLASS_NAMES = [ID_TO_ACTIVITY[i] for i in range(NUM_CLASSES)]
SHORT_NAMES = [
    "Wash hands", "Floor clean", "Window clean", "Vacuum",
    "Brush teeth", "Wash dishes", "Put away dishes", "Make tea",
    "Cut veg", "Cream hands", "Drink", "Water plants",
    "Clean table", "Disinfect", "Clean dishwasher",
    "Null",
]

FUSION_COLOR  = "#9C27B0"
FOLD_COLORS   = ["#1f77b4", "#ff7f0e", "#2ca02c"]
UNIMODAL_COLORS = {"TinyHAR": "#2196F3", "TSMixer": "#4CAF50", "AST": "#FF5722"}

sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 150})


# ── Model ─────────────────────────────────────────────────────────────────────

class LateFusionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = LateFusion(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, hidden_dim=256, dropout=0.3,
        )

    def forward(self, imu, audio_list, humidity):
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        return self.fusion(self.tinyhar(imu), self.ast(padded), self.tsmixer(humidity))


def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        logits   = model(imu, audio, humidity)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    elapsed = time.time() - t0
    return np.concatenate(all_preds), np.concatenate(all_labels), elapsed


def evaluate_all_folds(splits, device):
    fold_results = []
    cm_sum       = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for fold_idx, fold in enumerate(splits):
        ckpt = os.path.join(MODELS_DIR, f"late_fusion_fold{fold_idx}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  ⚠️  Missing: {ckpt} — skipping fold {fold_idx}")
            continue

        model = LateFusionModel().to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

        test_ds     = HARMESDataset(fold["test"], DATA_DIR,
                                    window_size=WINDOW_SIZE, step=STEP)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4,
                                 pin_memory=True)

        preds, labels, infer_time = run_inference(model, test_loader, device)

        f1           = f1_score(labels, preds, average="macro", zero_division=0)
        acc          = accuracy_score(labels, preds)
        per_class_f1 = f1_score(labels, preds, average=None,
                                labels=list(range(NUM_CLASSES)), zero_division=0)
        from sklearn.metrics import precision_score, recall_score
        per_class_prec = precision_score(labels, preds, average=None,
                                         labels=list(range(NUM_CLASSES)), zero_division=0)
        per_class_rec  = recall_score(labels, preds, average=None,
                                      labels=list(range(NUM_CLASSES)), zero_division=0)
        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
        cm_sum += cm

        fold_results.append({
            "fold":               fold_idx,
            "test_participants":  fold["test"],
            "train_participants": fold["train"],
            "n_windows":          len(test_ds),
            "macro_f1":           float(f1),
            "accuracy":           float(acc),
            "per_class_f1":       per_class_f1.tolist(),
            "per_class_precision": per_class_prec.tolist(),
            "per_class_recall":    per_class_rec.tolist(),
            "confusion_matrix":   cm.tolist(),
            "inference_time_s":   float(infer_time),
            "preds":              preds.tolist(),
            "labels":             labels.tolist(),
        })
        print(f"  Fold {fold_idx}: F1={f1:.4f}  acc={acc:.4f}  "
              f"test={fold['test']}  n={len(test_ds):,}")

    return fold_results, cm_sum


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_fold_log(fold_idx):
    pattern = f"logs/late_fusion_10s_fold{fold_idx}_*.out"
    matches = sorted(glob.glob(pattern))
    if not matches:
        return []
    log_path = matches[-1]
    epoch_re = re.compile(
        r"Epoch\s+(\d+)/\d+\s+loss=([\d.]+)\s+F1=([\d.]+)\s+acc=([\d.]+)"
    )
    epochs = []
    with open(log_path) as f:
        for line in f:
            if m := epoch_re.search(line):
                epochs.append({
                    "epoch": int(m.group(1)),
                    "loss":  float(m.group(2)),
                    "f1":    float(m.group(3)),
                    "acc":   float(m.group(4)),
                })
    return epochs


# ── Tables ────────────────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ {path}")


def save_tables(fold_results):
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Summary
    f1s  = [r["macro_f1"] for r in fold_results]
    accs = [r["accuracy"]  for r in fold_results]
    write_csv(
        os.path.join(TABLES_DIR, "late_fusion_summary.csv"),
        [{"model": "Late Fusion", "mean_f1": f"{np.mean(f1s):.4f}",
          "std_f1": f"{np.std(f1s):.4f}", "mean_acc": f"{np.mean(accs):.4f}",
          "std_acc": f"{np.std(accs):.4f}", "n_folds": len(fold_results)}],
        ["model", "mean_f1", "std_f1", "mean_acc", "std_acc", "n_folds"],
    )

    # Per-fold
    rows = []
    for r in fold_results:
        rows.append({
            "fold":              r["fold"],
            "test_participants": str(r["test_participants"]),
            "n_windows":         r["n_windows"],
            "macro_f1":          f"{r['macro_f1']:.4f}",
            "accuracy":          f"{r['accuracy']:.4f}",
            "inference_time_s":  f"{r['inference_time_s']:.1f}",
        })
    write_csv(
        os.path.join(TABLES_DIR, "late_fusion_per_fold.csv"),
        rows,
        ["fold", "test_participants", "n_windows", "macro_f1", "accuracy", "inference_time_s"],
    )

    # Per-class F1
    per_class = np.array([r["per_class_f1"] for r in fold_results])
    rows = []
    for i in range(NUM_CLASSES):
        rows.append({
            "class_id":   i,
            "class_name": CLASS_NAMES[i],
            "f1_mean":    f"{per_class[:, i].mean():.4f}",
            "f1_std":     f"{per_class[:, i].std():.4f}",
            "f1_fold0":   f"{per_class[0, i]:.4f}" if len(per_class) > 0 else "",
            "f1_fold1":   f"{per_class[1, i]:.4f}" if len(per_class) > 1 else "",
            "f1_fold2":   f"{per_class[2, i]:.4f}" if len(per_class) > 2 else "",
        })
    write_csv(
        os.path.join(TABLES_DIR, "late_fusion_per_class_f1.csv"),
        rows,
        ["class_id", "class_name", "f1_mean", "f1_std", "f1_fold0", "f1_fold1", "f1_fold2"],
    )

    # Most confused pairs
    cm_sum = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for r in fold_results:
        cm_sum += np.array(r["confusion_matrix"])
    cm_off = cm_sum.astype(float)
    np.fill_diagonal(cm_off, 0)
    indices = np.argsort(cm_off.flatten())[::-1][:10]
    rows = []
    for rank, idx in enumerate(indices, 1):
        tc = idx // NUM_CLASSES
        pc = idx  % NUM_CLASSES
        rows.append({
            "rank":       rank,
            "true_class": CLASS_NAMES[tc],
            "pred_class": CLASS_NAMES[pc],
            "count":      int(cm_off[tc, pc]),
        })
    write_csv(
        os.path.join(TABLES_DIR, "late_fusion_most_confused.csv"),
        rows,
        ["rank", "true_class", "pred_class", "count"],
    )


# ── Figure helpers ────────────────────────────────────────────────────────────

def _save(fig, directory, filename):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 1: Training curves ─────────────────────────────────────────────────

def fig_training_curves(log_data, out_dir):
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Late Fusion (10s) — Training Curves", fontweight="bold")

    for fold_idx, epochs in log_data.items():
        if not epochs:
            continue
        c  = FOLD_COLORS[fold_idx]
        xs = [e["epoch"] for e in epochs]
        ax_loss.plot(xs, [e["loss"] for e in epochs],
                     label=f"Fold {fold_idx}", color=c, linewidth=2)
        ax_f1.plot(xs, [e["f1"] for e in epochs],
                   label=f"Fold {fold_idx}", color=c, linewidth=2)
        best = max(epochs, key=lambda e: e["f1"])
        ax_f1.scatter([best["epoch"]], [best["f1"]], color=c, s=100,
                      zorder=5, marker="*")
        ax_f1.annotate(f"  {best['f1']:.3f}", (best["epoch"], best["f1"]),
                       fontsize=8, color=c)

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Training Loss"); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    sns.despine(ax=ax_loss)

    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Macro F1")
    ax_f1.set_title("Test Macro F1 per Epoch"); ax_f1.set_ylim(0, 1)
    ax_f1.axhline(1/NUM_CLASSES, color="gray", linestyle="--",
                  linewidth=1, label="Random chance")
    ax_f1.legend(); ax_f1.grid(True, alpha=0.3)
    sns.despine(ax=ax_f1)

    plt.tight_layout()
    _save(fig, out_dir, "training_curves.png")


# ── Figure 2: Per-fold F1 and Accuracy ───────────────────────────────────────

def fig_per_fold(fold_results, out_dir):
    folds   = [r["fold"] for r in fold_results]
    f1s     = [r["macro_f1"] for r in fold_results]
    accs    = [r["accuracy"]  for r in fold_results]
    mean_f1 = np.mean(f1s)
    std_f1  = np.std(f1s)

    xs    = np.arange(len(folds))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))

    bars_f1  = ax.bar(xs - width/2, f1s,  width, color=FUSION_COLOR,
                      alpha=0.85, label="Macro F1")
    bars_acc = ax.bar(xs + width/2, accs, width, color="#FF9800",
                      alpha=0.85, label="Accuracy")

    for bar, v in zip(bars_f1, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=FUSION_COLOR)
    for bar, v in zip(bars_acc, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#E65100")

    ax.axhline(mean_f1, color=FUSION_COLOR, linestyle="--", linewidth=1.5,
               label=f"Mean F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Fold {i}\n(test P{fold_results[i]['test_participants'][0]}–"
                        f"P{fold_results[i]['test_participants'][-1]})"
                        for i in folds], fontsize=10)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Late Fusion (10s) — Per-Fold Results\n"
                 "TinyHAR + AST + TSMixer → concat(384) → MLP → 15 classes",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "per_fold_f1.png")


# ── Figure 3: Per-class F1 (mean ± std) ──────────────────────────────────────

def fig_per_class_f1(fold_results, out_dir, suffix="", fold_idx=None):
    per_class = np.array([r["per_class_f1"] for r in fold_results])
    if fold_idx is not None:
        means = per_class[fold_idx]
        stds  = None
        title = f"Late Fusion (10s) — Per-Class F1, Fold {fold_idx}"
        fname = "per_class_f1.png"
    else:
        means = per_class.mean(axis=0)
        stds  = per_class.std(axis=0)
        title = "Late Fusion (10s) — Per-Class F1 (Mean ± Std, 3-Fold CV)"
        fname = "per_class_f1.png"

    xs  = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(15, 6))

    if stds is not None:
        ax.bar(xs, means, color=FUSION_COLOR, alpha=0.85,
               yerr=stds, capsize=4, error_kw={"linewidth": 1.5})
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.04, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90, fontweight="bold")
    else:
        ax.bar(xs, means, color=FUSION_COLOR, alpha=0.85)
        for x, m in zip(xs, means):
            ax.text(x, m + 0.03, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90, fontweight="bold")

    ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--",
               linewidth=1.2, label=f"Random ({1/NUM_CLASSES:.2f})")
    ax.axhline(np.mean(means), color=FUSION_COLOR, linestyle=":",
               linewidth=1.5, label=f"Mean F1 = {np.mean(means):.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.3)
    ax.set_title(title + suffix, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, fname)


# ── Figure 4: Confusion matrix ────────────────────────────────────────────────

def fig_confusion_matrix(cm, out_dir, title_suffix=""):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, ax=ax,
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                cmap="Purples", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor="lightgray",
                cbar_kws={"shrink": 0.8, "label": "Recall"})
    ax.set_title(f"Late Fusion (10s) — Confusion Matrix{title_suffix}",
                 fontweight="bold")
    ax.set_xlabel("Predicted Activity"); ax.set_ylabel("True Activity")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    _save(fig, out_dir, "confusion_matrix.png")


# ── Figure 5: Fusion vs Unimodal comparison ───────────────────────────────────

def fig_fusion_vs_unimodal(fold_results, unimodal_data, out_dir):
    models = ["TSMixer", "AST", "TinyHAR", "Late Fusion"]
    colors = [UNIMODAL_COLORS["TSMixer"], UNIMODAL_COLORS["AST"],
              UNIMODAL_COLORS["TinyHAR"], FUSION_COLOR]

    mean_f1s, std_f1s, mean_accs = [], [], []
    for name in ["TSMixer", "AST", "TinyHAR"]:
        folds = unimodal_data.get(name, [])
        f1s   = [r["macro_f1"] for r in folds]
        accs  = [r["accuracy"]  for r in folds]
        mean_f1s.append(np.mean(f1s)); std_f1s.append(np.std(f1s))
        mean_accs.append(np.mean(accs))

    lf_f1s  = [r["macro_f1"] for r in fold_results]
    lf_accs = [r["accuracy"]  for r in fold_results]
    mean_f1s.append(np.mean(lf_f1s));  std_f1s.append(np.std(lf_f1s))
    mean_accs.append(np.mean(lf_accs))

    xs = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.bar(xs, mean_f1s, 0.55, yerr=std_f1s, capsize=7,
                  color=colors, alpha=0.88,
                  error_kw={"linewidth": 2})

    for bar, m, s, a in zip(bars, mean_f1s, std_f1s, mean_accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.018,
                f"F1={m:.4f}\nAcc={a:.4f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--",
               linewidth=1.2, label=f"Random chance ({1/NUM_CLASSES:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Macro F1 Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Late Fusion vs Unimodal Baselines (10s Windows)\n"
                 "HARMES — 20 Participants — 3-Fold Group CV",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "fusion_vs_unimodal.png")


# ── Figure 6: Per-class comparison (Late Fusion vs TinyHAR vs AST) ───────────

def fig_per_class_comparison(fold_results, unimodal_data, out_dir):
    lf_means = np.array([r["per_class_f1"] for r in fold_results]).mean(axis=0)

    th_folds  = unimodal_data.get("TinyHAR", [])
    ast_folds = unimodal_data.get("AST", [])
    th_means  = np.array([r["per_class_f1"] for r in th_folds]).mean(axis=0) if th_folds else np.zeros(NUM_CLASSES)
    ast_means = np.array([r["per_class_f1"] for r in ast_folds]).mean(axis=0) if ast_folds else np.zeros(NUM_CLASSES)

    xs    = np.arange(NUM_CLASSES)
    width = 0.27
    fig, ax = plt.subplots(figsize=(17, 7))

    b1 = ax.bar(xs - width, th_means,  width, color=UNIMODAL_COLORS["TinyHAR"],
                alpha=0.8, label="TinyHAR (IMU)")
    b2 = ax.bar(xs,         ast_means, width, color=UNIMODAL_COLORS["AST"],
                alpha=0.8, label="AST (Audio)")
    b3 = ax.bar(xs + width, lf_means,  width, color=FUSION_COLOR,
                alpha=0.88, label="Late Fusion (All)")

    for bar, v in zip(b3, lf_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                rotation=90, color=FUSION_COLOR, fontweight="bold")

    ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Per-Class F1 — Late Fusion vs Best Unimodal Baselines (10s)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "per_class_comparison.png")


# ── Figure 7: Lift over best unimodal ────────────────────────────────────────

def fig_lift_over_unimodal(fold_results, unimodal_data, out_dir):
    lf_means = np.array([r["per_class_f1"] for r in fold_results]).mean(axis=0)

    # Best unimodal per class (element-wise max across TinyHAR, AST, TSMixer)
    best_uni = np.zeros(NUM_CLASSES)
    for name in ["TinyHAR", "AST", "TSMixer"]:
        folds = unimodal_data.get(name, [])
        if folds:
            arr = np.array([r["per_class_f1"] for r in folds]).mean(axis=0)
            best_uni = np.maximum(best_uni, arr)

    lift   = lf_means - best_uni
    colors = [FUSION_COLOR if v >= 0 else "#F44336" for v in lift]

    xs  = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(xs, lift, color=colors, alpha=0.85)

    for bar, v in zip(bars, lift):
        ypos = bar.get_height() + 0.005 if v >= 0 else bar.get_height() - 0.035
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:+.3f}", ha="center", va="bottom", fontsize=7.5,
                rotation=90, fontweight="bold",
                color=FUSION_COLOR if v >= 0 else "#F44336")

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(np.mean(lift), color="gray", linestyle="--", linewidth=1.5,
               label=f"Mean lift = {np.mean(lift):+.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Lift (Late Fusion − Best Unimodal)")
    ax.set_title("Per-Class F1 Lift — Late Fusion over Best Unimodal Baseline (10s)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "lift_over_best_unimodal.png")


# ── Figure 8: Window-size progression (2s → 8s → 10s → Fusion) ───────────────

def fig_progression(fold_results, out_dir):
    data = {
        "TinyHAR": [0.6447, 0.7363, 0.7491],
        "AST":     [0.6715, 0.6935, 0.6999],
        "TSMixer": [0.0805, 0.0940, 0.1149],
    }
    lf_mean = np.mean([r["macro_f1"] for r in fold_results])

    fig, ax = plt.subplots(figsize=(10, 6))
    xs      = [0, 1, 2, 3]
    xlabels = ["2s", "8s", "10s", "10s\nLate Fusion"]

    for name, vals in data.items():
        ax.plot([0, 1, 2], vals, marker="o", linewidth=2,
                color=UNIMODAL_COLORS[name], label=name)
        ax.text(2.05, vals[-1], f" {vals[-1]:.3f}", fontsize=8,
                color=UNIMODAL_COLORS[name], va="center")

    ax.scatter([3], [lf_mean], color=FUSION_COLOR, s=180, zorder=5,
               marker="*", label=f"Late Fusion ({lf_mean:.4f})")
    ax.annotate(f"  {lf_mean:.4f}", (3, lf_mean), fontsize=10,
                fontweight="bold", color=FUSION_COLOR, va="center")
    ax.plot([2, 3], [0.7491, lf_mean], linestyle="--",
            color=FUSION_COLOR, linewidth=1.5, alpha=0.6)

    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("Macro F1 Score"); ax.set_ylim(0, 1)
    ax.set_title("Macro F1 Progression: Window Size → Fusion (HARMES, 3-Fold CV)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "progression_to_fusion.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE/50:.0f}s\n")

    splits = get_3fold_splits(list(range(1, 21)))

    # ── 1. Inference ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Running inference from late fusion checkpoints...")
    print("=" * 60)
    fold_results, cm_sum = evaluate_all_folds(splits, device)

    if not fold_results:
        print("No fold results — check that checkpoints exist in", MODELS_DIR)
        return

    # Save JSON
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w") as f:
        clean = [{k: v for k, v in r.items() if k not in ("preds", "labels")}
                 for r in fold_results]
        json.dump(clean, f, indent=2)
    print(f"  ✓ {JSON_OUT}")

    # ── 2. Load unimodal results ──────────────────────────────────────────────
    unimodal_data = {}
    if os.path.exists(UNIMODAL_JSON):
        with open(UNIMODAL_JSON) as f:
            unimodal_data = json.load(f)
        print(f"  Loaded unimodal results from {UNIMODAL_JSON}")
    else:
        print(f"  ⚠️  {UNIMODAL_JSON} not found — comparison figures will be partial")

    # ── 3. Parse training logs ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Parsing training logs...")
    print("=" * 60)
    log_data = {}
    for fold_idx in range(3):
        epochs = parse_fold_log(fold_idx)
        log_data[fold_idx] = epochs
        print(f"  Fold {fold_idx}: {len(epochs)} epochs parsed")

    # ── 4. Tables ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving tables...")
    print("=" * 60)
    save_tables(fold_results)

    # ── 5. Figures ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\nTraining curves:")
    fig_training_curves(log_data, FIGURES_DIR)

    print("\nPer-fold results:")
    fig_per_fold(fold_results, FIGURES_DIR)

    print("\nPer-class F1 (averaged):")
    fig_per_class_f1(fold_results, FIGURES_DIR)

    print("\nAggregated confusion matrix:")
    fig_confusion_matrix(cm_sum, FIGURES_DIR, " (Aggregated across 3 Folds)")

    if unimodal_data:
        print("\nFusion vs unimodal comparison:")
        fig_fusion_vs_unimodal(fold_results, unimodal_data, FIGURES_DIR)

        print("\nPer-class comparison:")
        fig_per_class_comparison(fold_results, unimodal_data, FIGURES_DIR)

        print("\nLift over best unimodal:")
        fig_lift_over_unimodal(fold_results, unimodal_data, FIGURES_DIR)

    print("\nProgression figure:")
    fig_progression(fold_results, FIGURES_DIR)

    print("\nPer-fold figures:")
    for r in fold_results:
        fold_idx = r["fold"]
        fold_dir = os.path.join(FIGURES_DIR, f"fold{fold_idx}")
        cm = np.array(r["confusion_matrix"])
        fig_confusion_matrix(cm, fold_dir,
                             f" — Fold {fold_idx} (test=P{r['test_participants'][0]}–"
                             f"P{r['test_participants'][-1]})")
        fig_per_class_f1([r], fold_dir,
                         suffix=f" — Fold {fold_idx} (test={r['test_participants']})",
                         fold_idx=0)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    f1s  = [r["macro_f1"] for r in fold_results]
    accs = [r["accuracy"]  for r in fold_results]
    print("\n" + "=" * 60)
    print("LATE FUSION RESULTS — 10s Windows, 3-Fold CV")
    print("=" * 60)
    print(f"{'Fold':<8} {'F1':>8} {'Acc':>8}")
    print("-" * 28)
    for r in fold_results:
        print(f"  {r['fold']}      {r['macro_f1']:.4f}   {r['accuracy']:.4f}")
    print("-" * 28)
    print(f"  Mean    {np.mean(f1s):.4f}   {np.mean(accs):.4f}")
    print(f"  Std     {np.std(f1s):.4f}   {np.std(accs):.4f}")
    print(f"\nFigures → {FIGURES_DIR}/")
    print(f"Tables  → {TABLES_DIR}/")
    print(f"JSON    → {JSON_OUT}")


if __name__ == "__main__":
    main()
