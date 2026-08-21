"""
Modality Ablation Evaluation — IMU + Audio (2-mod) vs 3-mod, 10s windows.

Evaluates Late Fusion, GMF, and CMA trained WITHOUT the humidity modality.
Loads best checkpoints ({method}_2mod_fold{i}_best.pt), runs inference,
and produces per-method figures + cross-method ablation comparison figures
that directly quantify the humidity contribution.

Output structure:
    results/v3/10s/ablation/
    ├── late_fusion/
    │   ├── per_fold_f1.png
    │   ├── confusion_matrix.png
    │   ├── per_class_f1.png
    │   ├── 3mod_vs_2mod.png
    │   ├── tables/
    │   │   ├── summary.csv
    │   │   ├── per_fold.csv
    │   │   ├── per_class_f1.csv
    │   │   └── most_confused.csv
    │   ├── fold0/
    │   │   ├── confusion_matrix.png
    │   │   └── per_class_f1.png
    │   └── fold1/, fold2/  (same)
    ├── gmf/            (identical structure)
    ├── cma/            (identical structure)
    ├── ablation_summary.csv          ← 3-mod vs 2-mod for all methods
    ├── humidity_contribution.png     ← Δ F1 bar chart (3-mod − 2-mod)
    ├── ablation_per_fold.png         ← grouped bars: 3-mod vs 2-mod per fold
    ├── ablation_per_class.png        ← per-class F1 lines: 3-mod vs 2-mod
    └── ablation_leaderboard.png      ← lollipop: all 6 configs ranked

Usage (cluster):
    PYTHONPATH=. python jobs/evaluate_ablation_2mod_10s.py

Usage (local, no GPU):
    PYTHONPATH=. python jobs/evaluate_ablation_2mod_10s.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR   = os.path.join(ROOT, "results", "v3", "10s", "models")
OUT_DIR      = os.path.join(ROOT, "results", "v3", "10s", "ablation")
FUSION_JSON  = os.path.join(ROOT, "results", "v3", "10s", "fusion_results.json")

WINDOW_SIZE = 500
STEP        = 250
EMBED_DIM   = 128
BATCH_SIZE  = 64
NUM_WORKERS = 2

sys.path.insert(0, ROOT)

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.data.splits import get_3fold_splits
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.ast.model import ASTEncoder

# ---------------------------------------------------------------------------
# Activity labels
# ---------------------------------------------------------------------------
SHORT_NAMES = [
    "Wash hands",       # 0
    "Floor clean",      # 1
    "Window clean",     # 2
    "Vacuum",           # 3
    "Brush teeth",      # 4
    "Wash dishes",      # 5
    "Put away dishes",  # 6
    "Make tea",         # 7
    "Cut veg",          # 8
    "Cream hands",      # 9
    "Drink",            # 10
    "Water plants",     # 11
    "Clean table",      # 12
    "Disinfect",        # 13
    "Clean dishwasher", # 14
    "Null",             # 15
]

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
METHODS = {
    "late_fusion": {"name": "Late Fusion", "color": "#9C27B0"},
    "gmf":         {"name": "GMF",         "color": "#2196F3"},
    "cma":         {"name": "CMA",         "color": "#4CAF50"},
}

FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
FOLD_TEST_PARTICIPANTS = {0: "P1-P7", 1: "P8-P14", 2: "P15-P20"}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 150})

# ===========================================================================
# 2-Modality fusion modules  (must exactly match train_ablation_2mod_10s.py)
# ===========================================================================

class LateFusion2Mod(nn.Module):
    """Concat(IMU, Audio) → MLP → num_classes."""
    def __init__(self, imu_dim=128, audio_dim=128, num_classes=16,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(imu_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, imu_emb, audio_emb):
        return self.classifier(torch.cat([imu_emb, audio_emb], dim=-1))


class GMF2Mod(nn.Module):
    """Gated Multimodal Fusion — 2 modalities (IMU + Audio)."""
    def __init__(self, imu_dim=128, audio_dim=128, num_classes=16,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.gate_imu   = nn.Linear(imu_dim,   1)
        self.gate_audio = nn.Linear(audio_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(imu_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, imu_emb, audio_emb):
        g_imu   = torch.sigmoid(self.gate_imu(imu_emb))
        g_audio = torch.sigmoid(self.gate_audio(audio_emb))
        fused   = torch.cat([g_imu * imu_emb, g_audio * audio_emb], dim=-1)
        return self.classifier(fused)


class CMA2Mod(nn.Module):
    """Cross-Modal Attention — 2 modalities (IMU + Audio)."""
    def __init__(self, imu_dim=128, audio_dim=128, num_classes=16,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_imu_to_audio = nn.MultiheadAttention(
            embed_dim=imu_dim,   num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_audio_to_imu = nn.MultiheadAttention(
            embed_dim=audio_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_imu   = nn.LayerNorm(imu_dim)
        self.norm_audio = nn.LayerNorm(audio_dim)
        self.classifier = nn.Sequential(
            nn.Linear(imu_dim + audio_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, imu_emb, audio_emb):
        imu_q   = imu_emb.unsqueeze(1)
        audio_q = audio_emb.unsqueeze(1)
        imu_ctx,   _ = self.attn_imu_to_audio(imu_q,   audio_q, audio_q)
        audio_ctx, _ = self.attn_audio_to_imu(audio_q, imu_q,   imu_q)
        imu_out   = self.norm_imu(imu_emb     + imu_ctx.squeeze(1))
        audio_out = self.norm_audio(audio_emb + audio_ctx.squeeze(1))
        return self.classifier(torch.cat([imu_out, audio_out], dim=-1))


FUSION_CLASSES_2MOD = {
    "late_fusion": LateFusion2Mod,
    "gmf":         GMF2Mod,
    "cma":         CMA2Mod,
}

# ===========================================================================
# Full 2-mod model wrapper
# ===========================================================================
class FusionModel2Mod(nn.Module):
    """IMU + Audio only — no humidity/TSMixer pathway."""
    def __init__(self, fusion_module: nn.Module):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = fusion_module

    def forward(self, imu: torch.Tensor, audio_list: list) -> torch.Tensor:
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        return self.fusion(self.tinyhar(imu), self.ast(padded))

# ===========================================================================
# Data
# ===========================================================================
def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)

# ===========================================================================
# Inference
# ===========================================================================
@torch.no_grad()
def run_inference_fold(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()
    for imu, audio_list, _humidity, labels in loader:
        imu  = imu.to(device)
        out  = model(imu, audio_list)         # humidity ignored
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    elapsed    = time.time() - t0
    return np.concatenate(all_preds), np.concatenate(all_labels), elapsed


def run_inference_all_folds(key: str, splits: list, device: torch.device) -> list:
    """Load 2-mod checkpoints for all 3 folds and run inference."""
    fold_results = []
    for fold_i, split in enumerate(splits):
        ckpt_path = os.path.join(MODELS_DIR, f"{key}_2mod_fold{fold_i}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ✗ Checkpoint not found: {ckpt_path}")
            return []

        print(f"  Fold {fold_i}: loading {ckpt_path}")
        fusion_cls = FUSION_CLASSES_2MOD[key]
        model      = FusionModel2Mod(fusion_cls()).to(device)

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            for k in ("model_state_dict", "state_dict", "model_state"):
                if k in state:
                    state = state[k]
                    break
        model.load_state_dict(state)

        test_participants = split["test"]
        ds = HARMESDataset(
            participant_ids=test_participants,
            data_dir=DATA_DIR,
            window_size=WINDOW_SIZE,
            step=STEP,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

        preds, labels, inf_time = run_inference_fold(model, loader, device)

        macro_f1   = float(f1_score(labels, preds, average="macro", zero_division=0))
        accuracy   = float(accuracy_score(labels, preds))
        per_cls_f1 = f1_score(labels, preds, average=None, zero_division=0,
                               labels=list(range(NUM_CLASSES))).tolist()
        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist()

        fold_results.append({
            "fold":              fold_i,
            "test_participants": test_participants,
            "n_windows":         len(ds),
            "macro_f1":          macro_f1,
            "accuracy":          accuracy,
            "per_class_f1":      per_cls_f1,
            "confusion_matrix":  cm,
            "inference_time_s":  round(inf_time, 2),
        })
        print(f"    F1={macro_f1:.4f}  Acc={accuracy:.4f}  ({len(ds)} windows, {inf_time:.1f}s)")

        del model
        torch.cuda.empty_cache()

    return fold_results

# ===========================================================================
# Helpers
# ===========================================================================
def aggregate_cm(fold_results):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for r in fold_results:
        cm += np.array(r["confusion_matrix"])
    return cm

def row_normalise(cm):
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return cm / rs

def mean_per_class_f1(fold_results):
    arr = np.array([r["per_class_f1"] for r in fold_results])
    return arr.mean(axis=0), arr.std(axis=0)

def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {path}")

# ===========================================================================
# Tables
# ===========================================================================
def save_tables(fold_results, tables_dir, method_name, threemod_f1):
    f1s      = [r["macro_f1"] for r in fold_results]
    accs     = [r["accuracy"]  for r in fold_results]
    mean_f1  = float(np.mean(f1s));   std_f1  = float(np.std(f1s))
    mean_acc = float(np.mean(accs));  std_acc = float(np.std(accs))
    delta    = mean_f1 - threemod_f1

    # summary.csv
    with open(os.path.join(tables_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "modalities", "mean_f1", "std_f1", "mean_acc", "std_acc",
                    "n_folds", "delta_vs_3mod"])
        w.writerow([method_name, "IMU+Audio",
                    f"{mean_f1:.4f}", f"{std_f1:.4f}",
                    f"{mean_acc:.4f}", f"{std_acc:.4f}",
                    len(fold_results), f"{delta:+.4f}"])
    print(f"  ✓ summary.csv")

    # per_fold.csv
    with open(os.path.join(tables_dir, "per_fold.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "test_participants", "n_windows",
                    "macro_f1", "accuracy", "inference_time_s"])
        for r in fold_results:
            ps = r["test_participants"]
            w.writerow([r["fold"], f"P{min(ps)}-P{max(ps)}",
                        r["n_windows"], f"{r['macro_f1']:.4f}",
                        f"{r['accuracy']:.4f}", r["inference_time_s"]])
    print(f"  ✓ per_fold.csv")

    # per_class_f1.csv
    pc_means, pc_stds = mean_per_class_f1(fold_results)
    with open(os.path.join(tables_dir, "per_class_f1.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "f1_mean", "f1_std",
                    "f1_fold0", "f1_fold1", "f1_fold2"])
        for cls_id in range(NUM_CLASSES):
            row = [cls_id, SHORT_NAMES[cls_id],
                   f"{pc_means[cls_id]:.4f}", f"{pc_stds[cls_id]:.4f}"]
            for r in fold_results:
                row.append(f"{r['per_class_f1'][cls_id]:.4f}")
            w.writerow(row)
    print(f"  ✓ per_class_f1.csv")

    # most_confused.csv
    cm_agg = aggregate_cm(fold_results)
    np.fill_diagonal(cm_agg, 0)
    pairs = [(i, j, int(cm_agg[i, j])) for i in range(NUM_CLASSES)
             for j in range(NUM_CLASSES) if i != j and cm_agg[i, j] > 0]
    pairs.sort(key=lambda x: -x[2])
    with open(os.path.join(tables_dir, "most_confused.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "true_class", "pred_class", "count"])
        for rank, (tc, pc, cnt) in enumerate(pairs[:10], 1):
            w.writerow([rank, SHORT_NAMES[tc], SHORT_NAMES[pc], cnt])
    print(f"  ✓ most_confused.csv")

# ===========================================================================
# Per-method figures
# ===========================================================================

def fig_per_fold(fold_results, method_dir, cfg):
    n, x, w = len(fold_results), np.arange(len(fold_results)), 0.35
    f1s     = [r["macro_f1"] for r in fold_results]
    accs    = [r["accuracy"]  for r in fold_results]
    mean_f1 = float(np.mean(f1s))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_f1  = ax.bar(x - w/2, f1s,  w, color=cfg["color"], alpha=0.85, label="Macro F1")
    bars_acc = ax.bar(x + w/2, accs, w, color="#FF9800",     alpha=0.85, label="Accuracy")

    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(mean_f1, ls="--", color=cfg["color"], lw=1.5,
               label=f"Mean F1 = {mean_f1:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Fold {r['fold']}\n({FOLD_TEST_PARTICIPANTS[r['fold']]})"
        for r in fold_results
    ])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
    ax.set_title(f"{cfg['name']} (IMU+Audio) — Per-Fold Results",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(method_dir, "per_fold_f1.png"))


def fig_confusion_matrix_agg(fold_results, method_dir, cfg):
    cm_normed = row_normalise(aggregate_cm(fold_results))
    fig, ax   = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm_normed, ax=ax, cmap="Blues",
                annot=True, fmt=".2f", annot_kws={"size": 7},
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{cfg['name']} (IMU+Audio) — Confusion Matrix (Row-Normalised)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "confusion_matrix.png"))


def fig_per_class_f1_agg(fold_results, method_dir, cfg):
    means, stds = mean_per_class_f1(fold_results)
    mean_f1     = float(np.mean([r["macro_f1"] for r in fold_results]))
    x           = np.arange(NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=cfg["color"], alpha=0.85,
                  error_kw={"ecolor": "black", "lw": 1.2})

    for bar_obj, val, err in zip(bars, means, stds):
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2,
                val + err + 0.015, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(1/NUM_CLASSES, ls="--", color="grey",      lw=1.2, label="Random chance")
    ax.axhline(mean_f1,       ls=":",  color=cfg["color"], lw=1.5, label=f"Mean F1 = {mean_f1:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.18)
    ax.set_title(f"{cfg['name']} (IMU+Audio) — Per-Class F1 (Mean ± Std)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(method_dir, "per_class_f1.png"))


def fig_3mod_vs_2mod(fold_results_2mod, fold_results_3mod, method_dir, cfg):
    """Side-by-side bars: 3-mod vs 2-mod per fold + mean."""
    folds_label = [f"Fold {r['fold']}\n({FOLD_TEST_PARTICIPANTS[r['fold']]})"
                   for r in fold_results_2mod] + ["Mean"]

    f1_3 = [r["macro_f1"] for r in fold_results_3mod]
    f1_2 = [r["macro_f1"] for r in fold_results_2mod]
    f1_3.append(float(np.mean(f1_3)))
    f1_2.append(float(np.mean(f1_2)))

    x, w = np.arange(len(folds_label)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b3 = ax.bar(x - w/2, f1_3, w, color=cfg["color"], alpha=0.85, label="3-mod (IMU+Audio+Hum)")
    b2 = ax.bar(x + w/2, f1_2, w, color=cfg["color"], alpha=0.45, label="2-mod (IMU+Audio)", hatch="///")

    for bar, val in zip(list(b3) + list(b2), f1_3 + f1_2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    delta_mean = f1_2[-1] - f1_3[-1]
    ax.set_xticks(x)
    ax.set_xticklabels(folds_label)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.08)
    ax.set_title(
        f"{cfg['name']} — 3-Modality vs 2-Modality (Δ = {delta_mean:+.4f})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(method_dir, "3mod_vs_2mod.png"))


def fig_confusion_matrix_single(fold_r, fold_dir, cfg):
    cm_normed = row_normalise(np.array(fold_r["confusion_matrix"], dtype=float))
    fold_i    = fold_r["fold"]
    fig, ax   = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm_normed, ax=ax, cmap="Blues",
                annot=True, fmt=".2f", annot_kws={"size": 7},
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title(
        f"{cfg['name']} (2-mod) — Fold {fold_i} ({FOLD_TEST_PARTICIPANTS[fold_i]}) "
        f"Confusion Matrix",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(fold_dir, "confusion_matrix.png"))


def fig_per_class_f1_single(fold_r, fold_dir, cfg):
    f1s = fold_r["per_class_f1"]
    fold_i = fold_r["fold"]
    x      = np.arange(NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, f1s, color=cfg["color"], alpha=0.85)
    for bar_obj in bars:
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2,
                bar_obj.get_height() + 0.01, f"{bar_obj.get_height():.2f}",
                ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(1/NUM_CLASSES, ls="--", color="grey", lw=1.2, label="Random chance")
    ax.axhline(fold_r["macro_f1"], ls=":", color=cfg["color"], lw=1.5,
               label=f"Macro F1 = {fold_r['macro_f1']:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.18)
    ax.set_title(
        f"{cfg['name']} (2-mod) — Fold {fold_i} ({FOLD_TEST_PARTICIPANTS[fold_i]}) Per-Class F1",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(fold_dir, "per_class_f1.png"))

# ===========================================================================
# Cross-method ablation figures
# ===========================================================================

def fig_humidity_contribution(all_2mod, all_3mod, out_dir):
    """Bar chart: humidity Δ F1 per method (positive = humidity helps)."""
    methods, deltas, colors = [], [], []
    for key, cfg in METHODS.items():
        if key not in all_2mod or key not in all_3mod:
            continue
        f1_2 = float(np.mean([r["macro_f1"] for r in all_2mod[key]]))
        f1_3 = float(np.mean([r["macro_f1"] for r in all_3mod[key]]))
        delta = f1_3 - f1_2   # positive = humidity helps
        methods.append(cfg["name"])
        deltas.append(delta)
        colors.append("#43A047" if delta >= 0 else "#E53935")

    x   = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, deltas, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, deltas):
        va = "bottom" if val >= 0 else "top"
        off = 0.001 if val >= 0 else -0.001
        ax.text(bar.get_x() + bar.get_width()/2, val + off,
                f"{val:+.4f}", ha="center", va=va, fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("F1 Change from Adding Humidity\n(3-mod − 2-mod)")
    ax.set_title("Humidity Modality Contribution", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#43A047", alpha=0.85, label="Humidity helps"),
        Patch(facecolor="#E53935", alpha=0.85, label="Humidity hurts"),
    ], fontsize=9)
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(out_dir, "humidity_contribution.png"))


def fig_ablation_per_fold(all_2mod, all_3mod, out_dir):
    """Grouped bars: 3-mod vs 2-mod per fold for all methods."""
    keys   = [k for k in METHODS if k in all_2mod and k in all_3mod]
    n_keys = len(keys)
    n_folds = 3
    x       = np.arange(n_folds)
    width   = 0.35 / max(n_keys, 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, key in enumerate(keys):
        cfg    = METHODS[key]
        offset = (idx - n_keys / 2 + 0.5) * width

        f1_3 = [r["macro_f1"] for r in all_3mod[key]]
        f1_2 = [r["macro_f1"] for r in all_2mod[key]]

        bars_3 = ax.bar(x + offset - width/4, f1_3, width/2,
                        color=cfg["color"], alpha=0.85, label=f"{cfg['name']} 3-mod")
        bars_2 = ax.bar(x + offset + width/4, f1_2, width/2,
                        color=cfg["color"], alpha=0.40, hatch="///",
                        label=f"{cfg['name']} 2-mod")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}\n({FOLD_TEST_PARTICIPANTS[i]})"
                        for i in range(n_folds)])
    ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.08)
    ax.set_title("3-Mod vs 2-Mod (IMU+Audio) — Per Fold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation_per_fold.png"))


def fig_ablation_per_class(all_2mod, all_3mod, out_dir):
    """Line plot per-class F1: solid=3-mod, dashed=2-mod, one colour per method."""
    x    = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(16, 5))

    for key in METHODS:
        if key not in all_2mod or key not in all_3mod:
            continue
        cfg         = METHODS[key]
        means_3, _  = mean_per_class_f1(all_3mod[key])
        means_2, _  = mean_per_class_f1(all_2mod[key])
        ax.plot(x, means_3, color=cfg["color"], lw=1.8, marker="o", ms=4,
                label=f"{cfg['name']} 3-mod")
        ax.plot(x, means_2, color=cfg["color"], lw=1.2, ls="--", marker="s", ms=3,
                alpha=0.7, label=f"{cfg['name']} 2-mod")

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Macro F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 — 3-Mod vs 2-Mod (IMU+Audio)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation_per_class.png"))


def fig_ablation_leaderboard(all_2mod, all_3mod, out_dir):
    """Lollipop of all 6 configs (3 methods × 2 conditions) sorted by mean F1."""
    entries = []
    for key, cfg in METHODS.items():
        if key in all_3mod:
            f1s = [r["macro_f1"] for r in all_3mod[key]]
            entries.append({"name": f"{cfg['name']}\n(3-mod)", "mean_f1": float(np.mean(f1s)),
                             "std_f1": float(np.std(f1s)), "color": cfg["color"], "alpha": 0.9})
        if key in all_2mod:
            f1s = [r["macro_f1"] for r in all_2mod[key]]
            entries.append({"name": f"{cfg['name']}\n(2-mod)", "mean_f1": float(np.mean(f1s)),
                             "std_f1": float(np.std(f1s)), "color": cfg["color"], "alpha": 0.45})
    entries.sort(key=lambda e: e["mean_f1"])

    fig, ax = plt.subplots(figsize=(9, max(5, len(entries) * 1.1)))
    for i, e in enumerate(entries):
        ax.hlines(i, 0, e["mean_f1"], color=e["color"], lw=2.5, alpha=e["alpha"])
        ax.plot(e["mean_f1"], i, "o", color=e["color"], ms=11, alpha=e["alpha"])
        ax.errorbar(e["mean_f1"], i, xerr=e["std_f1"],
                    fmt="none", ecolor=e["color"], capsize=5, elinewidth=1.8, alpha=e["alpha"])
        ax.text(e["mean_f1"] + e["std_f1"] + 0.007, i,
                f"{e['mean_f1']:.4f} ± {e['std_f1']:.4f}", va="center", fontsize=9)

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e["name"] for e in entries], fontsize=10)
    ax.set_xlabel("Mean Macro F1 (3-fold CV)")
    ax.set_title("Ablation Leaderboard — 3-Mod vs 2-Mod",
                 fontsize=13, fontweight="bold")
    max_val = max(e["mean_f1"] + e["std_f1"] for e in entries) if entries else 1.0
    ax.set_xlim(0, min(max_val + 0.14, 1.05))
    sns.despine(); fig.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation_leaderboard.png"))


def save_ablation_summary_csv(all_2mod, all_3mod, out_dir):
    """Master CSV: one row per method, 3-mod vs 2-mod columns side by side."""
    path = os.path.join(out_dir, "ablation_summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "method",
            "f1_3mod", "std_3mod", "f1_2mod", "std_2mod",
            "humidity_delta",
            "fold0_3mod", "fold1_3mod", "fold2_3mod",
            "fold0_2mod", "fold1_2mod", "fold2_2mod",
        ])
        for key, cfg in METHODS.items():
            if key not in all_2mod or key not in all_3mod:
                continue
            f1_3 = [r["macro_f1"] for r in all_3mod[key]]
            f1_2 = [r["macro_f1"] for r in all_2mod[key]]
            m3, s3 = float(np.mean(f1_3)), float(np.std(f1_3))
            m2, s2 = float(np.mean(f1_2)), float(np.std(f1_2))
            delta  = m3 - m2
            w.writerow([
                cfg["name"],
                f"{m3:.4f}", f"{s3:.4f}", f"{m2:.4f}", f"{s2:.4f}",
                f"{delta:+.4f}",
                f"{f1_3[0]:.4f}", f"{f1_3[1]:.4f}", f"{f1_3[2]:.4f}",
                f"{f1_2[0]:.4f}", f"{f1_2[1]:.4f}", f"{f1_2[2]:.4f}",
            ])
    print(f"  ✓ {path}")

# ===========================================================================
# Load 3-mod results from JSON
# ===========================================================================
def load_3mod_results() -> dict:
    """
    Load 3-mod fold results from the fusion_results.json.
    Falls back to per-method JSON files if master doesn't exist.
    """
    if os.path.exists(FUSION_JSON):
        with open(FUSION_JSON) as fh:
            data = json.load(fh)
        # fusion_results.json may be keyed by method name or be a flat list
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in METHODS}

    # Try per-method files
    results = {}
    for key in METHODS:
        p = os.path.join(os.path.dirname(FUSION_JSON), f"{key}_results.json")
        if os.path.exists(p):
            with open(p) as fh:
                results[key] = json.load(fh)
    return results

# ===========================================================================
# Summary printout
# ===========================================================================
def print_summary(all_2mod, all_3mod):
    print("\n" + "=" * 72)
    print("ABLATION SUMMARY — 3-MOD vs 2-MOD (IMU+Audio)")
    print("=" * 72)
    print(f"{'Method':<16} {'3-mod F1':>10} {'2-mod F1':>10} {'Δ (hum)':>10}")
    print("-" * 50)
    for key, cfg in METHODS.items():
        if key not in all_2mod or key not in all_3mod:
            continue
        f1_3 = float(np.mean([r["macro_f1"] for r in all_3mod[key]]))
        f1_2 = float(np.mean([r["macro_f1"] for r in all_2mod[key]]))
        print(f"{cfg['name']:<16} {f1_3:>10.4f} {f1_2:>10.4f} {f1_3 - f1_2:>+10.4f}")
    print("=" * 72)
    print("Δ > 0 means humidity helps; Δ < 0 means humidity hurts.\n")

# ===========================================================================
# Main
# ===========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"Models dir:  {MODELS_DIR}")
    print(f"Output dir:  {OUT_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    splits = get_3fold_splits(list(range(1, 21)))

    # Load 3-mod results for comparison
    all_3mod = load_3mod_results()
    if not all_3mod:
        print("WARNING: No 3-mod results found — comparison figures will be skipped.")

    all_2mod: dict = {}

    # -----------------------------------------------------------------------
    # Per-method evaluation
    # -----------------------------------------------------------------------
    for key, cfg in METHODS.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {cfg['name']} (2-mod) ...")

        method_dir = os.path.join(OUT_DIR, key)
        tables_dir = os.path.join(method_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        fold_results = run_inference_all_folds(key, splits, device)
        if not fold_results:
            print(f"  Skipping {cfg['name']} — checkpoints not found.")
            continue

        all_2mod[key] = fold_results

        # Save JSON
        json_path = os.path.join(OUT_DIR, f"{key}_2mod_results.json")
        with open(json_path, "w") as fh:
            json.dump(fold_results, fh, indent=2)
        print(f"  ✓ {json_path}")

        # Tables
        threemod_f1 = (float(np.mean([r["macro_f1"] for r in all_3mod[key]]))
                       if key in all_3mod else 0.0)
        save_tables(fold_results, tables_dir, cfg["name"], threemod_f1)

        # Per-method figures
        fig_per_fold(fold_results, method_dir, cfg)
        fig_confusion_matrix_agg(fold_results, method_dir, cfg)
        fig_per_class_f1_agg(fold_results, method_dir, cfg)

        if key in all_3mod:
            fig_3mod_vs_2mod(fold_results, all_3mod[key], method_dir, cfg)

        for fold_r in fold_results:
            fold_dir = os.path.join(method_dir, f"fold{fold_r['fold']}")
            os.makedirs(fold_dir, exist_ok=True)
            fig_confusion_matrix_single(fold_r, fold_dir, cfg)
            fig_per_class_f1_single(fold_r, fold_dir, cfg)

    # -----------------------------------------------------------------------
    # Cross-method ablation figures
    # -----------------------------------------------------------------------
    if len(all_2mod) >= 2:
        print(f"\n{'=' * 60}")
        print("Generating cross-method ablation figures ...")
        fig_humidity_contribution(all_2mod, all_3mod, OUT_DIR)
        fig_ablation_per_fold(all_2mod, all_3mod, OUT_DIR)
        fig_ablation_per_class(all_2mod, all_3mod, OUT_DIR)
        fig_ablation_leaderboard(all_2mod, all_3mod, OUT_DIR)
        save_ablation_summary_csv(all_2mod, all_3mod, OUT_DIR)

    if all_2mod:
        print_summary(all_2mod, all_3mod)

    print("Done.")


if __name__ == "__main__":
    main()
