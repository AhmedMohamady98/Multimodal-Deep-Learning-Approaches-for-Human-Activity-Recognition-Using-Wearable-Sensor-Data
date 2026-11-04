"""
CMA 10s evaluation script.

Loads cma_fold{0,1,2}_best.pt, runs inference, generates comprehensive
high-quality figures (300 DPI, seaborn ticks, value labels on every bar) and
tables. Figures compare CMA against unimodal baselines, Late Fusion, GMF, and MBT.

Output structure:
    results/cma/figures/
    ├── training_curves.png
    ├── per_fold_f1.png
    ├── per_class_f1.png
    ├── confusion_matrix.png
    ├── fusion_vs_unimodal.png
    ├── per_class_comparison.png
    ├── lift_over_best_unimodal.png
    ├── cma_vs_late_fusion.png
    ├── cma_vs_mbt.png
    ├── progression_to_cma.png
    ├── fold0/ {confusion_matrix, per_class_f1}.png
    ├── fold1/ ...
    └── fold2/ ...

    results/cma/tables/
    ├── cma_summary.csv
    ├── cma_per_fold.csv
    ├── cma_per_class_f1.csv
    └── cma_most_confused.csv

    results/cma/cma_results.json

Usage:
    PYTHONPATH=. python jobs/evaluate_cma_10s.py
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
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                              precision_score, recall_score)

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.data.splits import get_3fold_splits
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.cma import CrossModalAttention

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR     = "results/models"
FIGURES_DIR    = "results/figures/cma"
TABLES_DIR     = "results/tables"
JSON_OUT       = "results/cma_results.json"
UNIMODAL_JSON  = "results/unimodal_results.json"
LF_JSON        = "results/late_fusion_results.json"
GMF_JSON       = "results/gmf_results.json"
MBT_JSON       = "results/mbt_results.json"

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

FUSION_COLOR    = "#E65100"   # deep orange — CMA
LF_COLOR        = "#9C27B0"   # purple — Late Fusion
GMF_COLOR       = "#00897B"   # teal — GMF
MBT_COLOR       = "#3F51B5"   # indigo — MBT
FOLD_COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c"]
UNIMODAL_COLORS = {"TinyHAR": "#2196F3", "TSMixer": "#4CAF50", "AST": "#FF5722"}

sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 150})


# ── Model ─────────────────────────────────────────────────────────────────────

class CMAModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = CrossModalAttention(
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
        ckpt = os.path.join(MODELS_DIR, f"cma_fold{fold_idx}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  ⚠️  Missing: {ckpt} — skipping fold {fold_idx}")
            continue

        model = CMAModel().to(device)
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
        per_class_prec = precision_score(labels, preds, average=None,
                                         labels=list(range(NUM_CLASSES)), zero_division=0)
        per_class_rec  = recall_score(labels, preds, average=None,
                                      labels=list(range(NUM_CLASSES)), zero_division=0)
        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
        cm_sum += cm

        fold_results.append({
            "fold":                fold_idx,
            "test_participants":   fold["test"],
            "train_participants":  fold["train"],
            "n_windows":           len(test_ds),
            "macro_f1":            float(f1),
            "accuracy":            float(acc),
            "per_class_f1":        per_class_f1.tolist(),
            "per_class_precision": per_class_prec.tolist(),
            "per_class_recall":    per_class_rec.tolist(),
            "confusion_matrix":    cm.tolist(),
            "inference_time_s":    float(infer_time),
            "preds":               preds.tolist(),
            "labels":              labels.tolist(),
        })
        print(f"  Fold {fold_idx}: F1={f1:.4f}  acc={acc:.4f}  "
              f"test={fold['test']}  n={len(test_ds):,}")

    return fold_results, cm_sum


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_fold_log(fold_idx):
    pattern = f"logs/cma_10s_fold{fold_idx}_*.out"
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

    f1s  = [r["macro_f1"] for r in fold_results]
    accs = [r["accuracy"]  for r in fold_results]
    write_csv(
        os.path.join(TABLES_DIR, "cma_summary.csv"),
        [{"model": "CMA", "mean_f1": f"{np.mean(f1s):.4f}",
          "std_f1": f"{np.std(f1s):.4f}", "mean_acc": f"{np.mean(accs):.4f}",
          "std_acc": f"{np.std(accs):.4f}", "n_folds": len(fold_results)}],
        ["model", "mean_f1", "std_f1", "mean_acc", "std_acc", "n_folds"],
    )

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
        os.path.join(TABLES_DIR, "cma_per_fold.csv"),
        rows,
        ["fold", "test_participants", "n_windows", "macro_f1", "accuracy", "inference_time_s"],
    )

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
        os.path.join(TABLES_DIR, "cma_per_class_f1.csv"),
        rows,
        ["class_id", "class_name", "f1_mean", "f1_std", "f1_fold0", "f1_fold1", "f1_fold2"],
    )

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
        os.path.join(TABLES_DIR, "cma_most_confused.csv"),
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
    fig.suptitle("CMA (10s) — Training Curves", fontweight="bold")

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
    ax.set_title("CMA (10s) — Per-Fold Results\n"
                 "Bidirectional Cross-Modal Attention — TinyHAR + AST + TSMixer",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "per_fold_f1.png")


# ── Figure 3: Per-class F1 ────────────────────────────────────────────────────

def fig_per_class_f1(fold_results, out_dir, suffix="", fold_idx=None):
    per_class = np.array([r["per_class_f1"] for r in fold_results])
    if fold_idx is not None:
        means = per_class[fold_idx]
        stds  = None
        title = f"CMA (10s) — Per-Class F1, Fold {fold_idx}"
    else:
        means = per_class.mean(axis=0)
        stds  = per_class.std(axis=0)
        title = "CMA (10s) — Per-Class F1 (Mean ± Std, 3-Fold CV)"

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
    _save(fig, out_dir, "per_class_f1.png")


# ── Figure 4: Confusion matrix ────────────────────────────────────────────────

def fig_confusion_matrix(cm, out_dir, title_suffix=""):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, ax=ax,
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                cmap="Oranges", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor="lightgray",
                cbar_kws={"shrink": 0.8, "label": "Recall"})
    ax.set_title(f"CMA (10s) — Confusion Matrix{title_suffix}",
                 fontweight="bold")
    ax.set_xlabel("Predicted Activity"); ax.set_ylabel("True Activity")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    _save(fig, out_dir, "confusion_matrix.png")


# ── Figure 5: Fusion vs Unimodal comparison ───────────────────────────────────

def fig_fusion_vs_unimodal(fold_results, unimodal_data, out_dir):
    models = ["TSMixer", "AST", "TinyHAR", "CMA"]
    colors = [UNIMODAL_COLORS["TSMixer"], UNIMODAL_COLORS["AST"],
              UNIMODAL_COLORS["TinyHAR"], FUSION_COLOR]

    mean_f1s, std_f1s, mean_accs = [], [], []
    for name in ["TSMixer", "AST", "TinyHAR"]:
        folds = unimodal_data.get(name, [])
        f1s   = [r["macro_f1"] for r in folds]
        accs  = [r["accuracy"]  for r in folds]
        mean_f1s.append(np.mean(f1s)); std_f1s.append(np.std(f1s))
        mean_accs.append(np.mean(accs))

    cma_f1s  = [r["macro_f1"] for r in fold_results]
    cma_accs = [r["accuracy"]  for r in fold_results]
    mean_f1s.append(np.mean(cma_f1s));  std_f1s.append(np.std(cma_f1s))
    mean_accs.append(np.mean(cma_accs))

    xs = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.bar(xs, mean_f1s, 0.55, yerr=std_f1s, capsize=7,
                  color=colors, alpha=0.88, error_kw={"linewidth": 2})

    for bar, m, s, a in zip(bars, mean_f1s, std_f1s, mean_accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.018,
                f"F1={m:.4f}\nAcc={a:.4f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--",
               linewidth=1.2, label=f"Random chance ({1/NUM_CLASSES:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Macro F1 Score"); ax.set_ylim(0, 1.1)
    ax.set_title("CMA vs Unimodal Baselines (10s Windows)\n"
                 "HARMES — 20 Participants — 3-Fold Group CV",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "fusion_vs_unimodal.png")


# ── Figure 6: Per-class comparison ───────────────────────────────────────────

def fig_per_class_comparison(fold_results, unimodal_data, out_dir):
    cma_means = np.array([r["per_class_f1"] for r in fold_results]).mean(axis=0)

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
    b3 = ax.bar(xs + width, cma_means, width, color=FUSION_COLOR,
                alpha=0.88, label="CMA (All)")

    for bar, v in zip(b3, cma_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                rotation=90, color=FUSION_COLOR, fontweight="bold")

    ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Per-Class F1 — CMA vs Best Unimodal Baselines (10s)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "per_class_comparison.png")


# ── Figure 7: Lift over best unimodal ────────────────────────────────────────

def fig_lift_over_unimodal(fold_results, unimodal_data, out_dir):
    cma_means = np.array([r["per_class_f1"] for r in fold_results]).mean(axis=0)

    best_uni = np.zeros(NUM_CLASSES)
    for name in ["TinyHAR", "AST", "TSMixer"]:
        folds = unimodal_data.get(name, [])
        if folds:
            arr = np.array([r["per_class_f1"] for r in folds]).mean(axis=0)
            best_uni = np.maximum(best_uni, arr)

    lift   = cma_means - best_uni
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
    ax.set_ylabel("F1 Lift (CMA − Best Unimodal)")
    ax.set_title("Per-Class F1 Lift — CMA over Best Unimodal Baseline (10s)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "lift_over_best_unimodal.png")


# ── Figure 8: CMA vs Late Fusion ─────────────────────────────────────────────

def fig_cma_vs_late_fusion(fold_results, lf_data, out_dir):
    if not lf_data:
        return
    cma_f1s = [r["macro_f1"] for r in fold_results]
    lf_f1s  = [r["macro_f1"] for r in lf_data]

    n     = min(len(cma_f1s), len(lf_f1s))
    xs    = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars_lf  = ax.bar(xs - width/2, lf_f1s[:n],  width, color=LF_COLOR,
                      alpha=0.85, label=f"Late Fusion (mean={np.mean(lf_f1s):.4f})")
    bars_cma = ax.bar(xs + width/2, cma_f1s[:n], width, color=FUSION_COLOR,
                      alpha=0.85, label=f"CMA (mean={np.mean(cma_f1s):.4f})")

    for bar, v in zip(bars_lf, lf_f1s[:n]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=LF_COLOR)
    for bar, v in zip(bars_cma, cma_f1s[:n]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=FUSION_COLOR)

    ax.axhline(np.mean(lf_f1s),  color=LF_COLOR,    linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axhline(np.mean(cma_f1s), color=FUSION_COLOR, linestyle="--", linewidth=1.2, alpha=0.6)

    fold_labels = [f"Fold {r['fold']}\n(P{r['test_participants'][0]}–P{r['test_participants'][-1]})"
                   for r in fold_results[:n]]
    ax.set_xticks(xs); ax.set_xticklabels(fold_labels, fontsize=10)
    ax.set_ylabel("Macro F1 Score"); ax.set_ylim(0.75, 0.95)
    ax.set_title("CMA vs Late Fusion — Per-Fold Macro F1 (10s)\n"
                 "Direct cross-modal attention vs simple concatenation",
                 fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "cma_vs_late_fusion.png")


# ── Figure 9: CMA vs MBT — per-fold direct comparison ────────────────────────

def fig_cma_vs_mbt(fold_results, mbt_data, out_dir):
    if not mbt_data:
        return
    cma_f1s = [r["macro_f1"] for r in fold_results]
    mbt_f1s = [r["macro_f1"] for r in mbt_data]

    n     = min(len(cma_f1s), len(mbt_f1s))
    xs    = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars_mbt = ax.bar(xs - width/2, mbt_f1s[:n], width, color=MBT_COLOR,
                      alpha=0.85, label=f"MBT (mean={np.mean(mbt_f1s):.4f})")
    bars_cma = ax.bar(xs + width/2, cma_f1s[:n], width, color=FUSION_COLOR,
                      alpha=0.85, label=f"CMA (mean={np.mean(cma_f1s):.4f})")

    for bar, v in zip(bars_mbt, mbt_f1s[:n]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=MBT_COLOR)
    for bar, v in zip(bars_cma, cma_f1s[:n]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=FUSION_COLOR)

    ax.axhline(np.mean(mbt_f1s), color=MBT_COLOR,    linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axhline(np.mean(cma_f1s), color=FUSION_COLOR, linestyle="--", linewidth=1.2, alpha=0.6)

    fold_labels = [f"Fold {r['fold']}\n(P{r['test_participants'][0]}–P{r['test_participants'][-1]})"
                   for r in fold_results[:n]]
    ax.set_xticks(xs); ax.set_xticklabels(fold_labels, fontsize=10)
    ax.set_ylabel("Macro F1 Score"); ax.set_ylim(0.75, 0.95)
    ax.set_title("CMA vs MBT — Per-Fold Macro F1 (10s)\n"
                 "Bidirectional cross-attention (6 modules) vs bottleneck tokens",
                 fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "cma_vs_mbt.png")


# ── Figure 10: Progression ────────────────────────────────────────────────────

def fig_progression(fold_results, lf_data, gmf_data, mbt_data, out_dir):
    unimodal_vals = {
        "TinyHAR": [0.6447, 0.7363, 0.7491],
        "AST":     [0.6715, 0.6935, 0.6999],
        "TSMixer": [0.0805, 0.0940, 0.1149],
    }
    cma_mean = np.mean([r["macro_f1"] for r in fold_results])
    lf_mean  = np.mean([r["macro_f1"] for r in lf_data])  if lf_data  else None
    gmf_mean = np.mean([r["macro_f1"] for r in gmf_data]) if gmf_data else None
    mbt_mean = np.mean([r["macro_f1"] for r in mbt_data]) if mbt_data else None

    fig, ax = plt.subplots(figsize=(14, 6))
    xlabels = ["2s", "8s", "10s", "10s\nLate Fusion", "10s\nGMF", "10s\nMBT", "10s\nCMA"]
    xs_uni  = [0, 1, 2]

    for name, vals in unimodal_vals.items():
        ax.plot(xs_uni, vals, marker="o", linewidth=2,
                color=UNIMODAL_COLORS[name], label=name)
        ax.text(2.05, vals[-1], f" {vals[-1]:.3f}", fontsize=8,
                color=UNIMODAL_COLORS[name], va="center")

    prev_x, prev_y = 2, 0.7491
    checkpoints = [
        (lf_mean,  3, LF_COLOR,     "D", "Late Fusion"),
        (gmf_mean, 4, GMF_COLOR,    "s", "GMF"),
        (mbt_mean, 5, MBT_COLOR,    "^", "MBT"),
    ]
    for mean_val, xpos, color, marker, label in checkpoints:
        if mean_val is not None:
            ax.plot([prev_x, xpos], [prev_y, mean_val], linestyle="--",
                    color=color, linewidth=1.2, alpha=0.5)
            ax.scatter([xpos], [mean_val], color=color, s=150, zorder=5,
                       marker=marker, label=f"{label} ({mean_val:.4f})")
            ax.annotate(f"  {mean_val:.4f}", (xpos, mean_val), fontsize=9,
                        color=color, va="center")
            prev_x, prev_y = xpos, mean_val

    ax.plot([prev_x, 6], [prev_y, cma_mean], linestyle="--",
            color=FUSION_COLOR, linewidth=1.5, alpha=0.7)
    ax.scatter([6], [cma_mean], color=FUSION_COLOR, s=220, zorder=5,
               marker="*", label=f"CMA ({cma_mean:.4f})")
    ax.annotate(f"  {cma_mean:.4f}", (6, cma_mean), fontsize=11,
                fontweight="bold", color=FUSION_COLOR, va="center")

    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Macro F1 Score"); ax.set_ylim(0, 1)
    ax.set_title("Macro F1 Progression: Window Size → All Fusion Methods\n"
                 "(HARMES, 3-Fold Group CV)", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, out_dir, "progression_to_cma.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE/50:.0f}s\n")

    splits = get_3fold_splits(list(range(1, 21)))

    print("=" * 60)
    print("Running inference from CMA checkpoints...")
    print("=" * 60)
    fold_results, cm_sum = evaluate_all_folds(splits, device)

    if not fold_results:
        print("No fold results — check that checkpoints exist in", MODELS_DIR)
        return

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w") as f:
        clean = [{k: v for k, v in r.items() if k not in ("preds", "labels")}
                 for r in fold_results]
        json.dump(clean, f, indent=2)
    print(f"  ✓ {JSON_OUT}")

    unimodal_data = {}
    if os.path.exists(UNIMODAL_JSON):
        with open(UNIMODAL_JSON) as f:
            unimodal_data = json.load(f)

    lf_data  = []
    if os.path.exists(LF_JSON):
        with open(LF_JSON) as f:
            lf_data = json.load(f)

    gmf_data = []
    if os.path.exists(GMF_JSON):
        with open(GMF_JSON) as f:
            gmf_data = json.load(f)

    mbt_data = []
    if os.path.exists(MBT_JSON):
        with open(MBT_JSON) as f:
            mbt_data = json.load(f)

    print("\n" + "=" * 60)
    print("Parsing training logs...")
    print("=" * 60)
    log_data = {}
    for fold_idx in range(3):
        epochs = parse_fold_log(fold_idx)
        log_data[fold_idx] = epochs
        print(f"  Fold {fold_idx}: {len(epochs)} epochs parsed")

    print("\n" + "=" * 60)
    print("Saving tables...")
    print("=" * 60)
    save_tables(fold_results)

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

    print("\nCMA vs Late Fusion comparison:")
    fig_cma_vs_late_fusion(fold_results, lf_data, FIGURES_DIR)

    print("\nCMA vs MBT comparison:")
    fig_cma_vs_mbt(fold_results, mbt_data, FIGURES_DIR)

    print("\nProgression figure:")
    fig_progression(fold_results, lf_data, gmf_data, mbt_data, FIGURES_DIR)

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

    f1s  = [r["macro_f1"] for r in fold_results]
    accs = [r["accuracy"]  for r in fold_results]
    print("\n" + "=" * 60)
    print("CMA RESULTS — 10s Windows, 3-Fold CV")
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
