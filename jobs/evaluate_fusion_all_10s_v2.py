"""
Fusion evaluation — all 4 methods, 10s windows, v2 (authoritative).

Evaluates Late Fusion, GMF, MBT, and CMA across 3 cross-validation folds.
Loads best checkpoints, runs inference, generates per-method and cross-method
figures, writes tables, and saves JSON results.

Unimodal baselines are loaded from results/v3/10s/unimodal_results.json
to produce comparison and lift figures.

Output structure:
    results/v3/10s/figures/
    ├── late_fusion/
    │   ├── training_curves.png
    │   ├── per_fold_f1.png
    │   ├── confusion_matrix.png
    │   ├── per_class_f1.png
    │   ├── fusion_vs_unimodal.png
    │   ├── per_class_comparison.png
    │   ├── lift_over_unimodal.png
    │   ├── tables/
    │   │   ├── summary.csv
    │   │   ├── per_fold.csv
    │   │   ├── per_class_f1.csv
    │   │   └── most_confused.csv
    │   ├── fold0/
    │   │   ├── confusion_matrix.png
    │   │   └── per_class_f1.png
    │   ├── fold1/  (same)
    │   └── fold2/  (same)
    ├── gmf/            (identical structure)
    ├── mbt/            (identical structure)
    ├── cma/            (identical structure)
    ├── fusion_method_comparison.png
    ├── fusion_per_fold_all_methods.png
    ├── fusion_per_class_all_methods.png
    └── fusion_leaderboard.png

    results/v3/10s/
    ├── late_fusion_results.json
    ├── gmf_results.json
    ├── mbt_results.json
    └── cma_results.json
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import json
import time
import re
import csv

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
MODELS_DIR    = os.path.join(ROOT, "results", "v3", "10s", "models")
FIGURES_BASE  = os.path.join(ROOT, "results", "v3", "10s", "figures")
RESULTS_BASE  = os.path.join(ROOT, "results", "v3", "10s")
UNIMODAL_JSON = os.path.join(ROOT, "results", "v3", "10s", "unimodal_results.json")
LOGS_DIR      = os.path.join(ROOT, "logs")

WINDOW_SIZE = 500
STEP        = 250
EMBED_DIM   = 128
BATCH_SIZE  = 64
NUM_WORKERS = 4

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
    "late_fusion":     {"name": "Late Fusion",     "color": "#9C27B0", "ckpt": "late_fusion"},
    "gmf":             {"name": "GMF",             "color": "#2196F3", "ckpt": "gmf"},
    "mbt":             {"name": "MBT",             "color": "#FF5722", "ckpt": "mbt"},
    "cma":             {"name": "CMA",             "color": "#4CAF50", "ckpt": "cma"},
    "cls_transformer": {"name": "CLS Transformer", "color": "#795548", "ckpt": "cls_transformer"},
    "lmf":             {"name": "LMF",             "color": "#F44336", "ckpt": "lmf"},
    "decision_fusion": {"name": "Decision Fusion", "color": "#FF9800", "ckpt": "decision_fusion"},
}

UNIMODAL_COLORS = {"TinyHAR": "#2196F3", "AST": "#FF5722", "TSMixer": "#4CAF50"}
FOLD_COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c"]

FOLD_TEST_PARTICIPANTS = {
    0: "P1-P7",
    1: "P8-P14",
    2: "P15-P20",
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 150})

# ---------------------------------------------------------------------------
# Source imports
# ---------------------------------------------------------------------------
sys.path.insert(0, ROOT)

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.data.splits import get_3fold_splits
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.late_fusion import LateFusion
from src.fusion.gmf import GatedMultimodalFusion
from src.fusion.mbt import MultimodalBottleneckTransformer
from src.fusion.cma import CrossModalAttention
from src.fusion.cls_transformer import CLSTransformerFusion
from src.fusion.lmf import LowRankMultimodalFusion
from src.fusion.decision_fusion import DecisionFusion

# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------
def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)

# ---------------------------------------------------------------------------
# Fusion module factory
# ---------------------------------------------------------------------------
def make_fusion_module(key: str) -> nn.Module:
    if key == "late_fusion":
        return LateFusion(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES, hidden_dim=256, dropout=0.3)
    elif key == "gmf":
        return GatedMultimodalFusion(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES, hidden_dim=256, dropout=0.3)
    elif key == "mbt":
        return MultimodalBottleneckTransformer(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES)
    elif key == "cma":
        return CrossModalAttention(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES)
    elif key == "cls_transformer":
        return CLSTransformerFusion(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, model_dim=256, num_heads=8,
            num_layers=4, dropout=0.1,
        )
    elif key == "lmf":
        return LowRankMultimodalFusion(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, dropout=0.3,
        )
    elif key == "decision_fusion":
        return DecisionFusion(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
        )
    else:
        raise ValueError(f"Unknown fusion key: {key}")

# ---------------------------------------------------------------------------
# Full fusion model wrapper
# ---------------------------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, fusion_module: nn.Module):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = fusion_module

    def forward(self, imu: torch.Tensor, audio_list: list, humidity: torch.Tensor) -> torch.Tensor:
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        imu_emb = self.tinyhar(imu)
        aud_emb = self.ast(padded)
        hum_emb = self.tsmixer(humidity)
        return self.fusion(imu_emb, aud_emb, hum_emb)

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
_LOG_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+loss=([\d.]+)\s+F1=([\d.]+)\s+acc=([\d.]+)"
)

def parse_log(ckpt_key: str, fold: int) -> dict:
    """Parse training log for a method/fold. Returns dict with lists or empty dict."""
    pattern = os.path.join(LOGS_DIR, f"{ckpt_key}_10s_fold{fold}_*.out")
    files   = sorted(glob.glob(pattern))
    if not files:
        return {}
    log_file = files[-1]
    epochs, losses, f1s, accs = [], [], [], []
    try:
        with open(log_file) as fh:
            for line in fh:
                m = _LOG_RE.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
                    f1s.append(float(m.group(3)))
                    accs.append(float(m.group(4)))
    except Exception:
        return {}
    if not epochs:
        return {}
    return {"epochs": epochs, "losses": losses, "f1s": f1s, "accs": accs}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference_fold(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run inference on one fold. Returns (preds, labels, inference_time_s)."""
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for imu, audio_list, humidity, labels in loader:
            imu      = imu.to(device)
            humidity = humidity.to(device)
            logits   = model(imu, audio_list, humidity)
            preds    = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    elapsed    = time.time() - t0
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels, elapsed


def run_inference_all_folds(
    key: str,
    cfg: dict,
    splits: list,
    device: torch.device,
) -> list:
    """Evaluate all 3 folds for one fusion method. Returns list of fold result dicts."""
    ckpt_key     = cfg["ckpt"]
    fold_results = []

    for fold_i, split in enumerate(splits):
        # Try _best suffix first, then plain
        ckpt_path = os.path.join(MODELS_DIR, f"{ckpt_key}_fold{fold_i}_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(MODELS_DIR, f"{ckpt_key}_fold{fold_i}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            return []

        print(f"  Fold {fold_i}: loading {ckpt_path}")

        fusion_mod = make_fusion_module(key)
        model      = FusionModel(fusion_mod).to(device)

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state)

        test_participants = split["test"]
        dataset = HARMESDataset(
            participant_ids=test_participants,
            data_dir=DATA_DIR,
            window_size=WINDOW_SIZE,
            step=STEP,
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
        )

        preds, labels, inf_time = run_inference_fold(model, loader, device)

        macro_f1   = float(f1_score(labels, preds, average="macro", zero_division=0))
        accuracy   = float(accuracy_score(labels, preds))
        per_cls_f1 = f1_score(
            labels, preds, average=None, zero_division=0,
            labels=list(range(NUM_CLASSES))
        ).tolist()
        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist()

        fold_results.append({
            "fold":              fold_i,
            "test_participants": test_participants,
            "n_windows":         len(dataset),
            "macro_f1":          macro_f1,
            "accuracy":          accuracy,
            "per_class_f1":      per_cls_f1,
            "confusion_matrix":  cm,
            "inference_time_s":  round(inf_time, 2),
        })

        print(f"    F1={macro_f1:.4f}  Acc={accuracy:.4f}"
              f"  ({len(dataset)} windows, {inf_time:.1f}s)")

    return fold_results

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def save_json(fold_results: list, key: str):
    out_path = os.path.join(RESULTS_BASE, f"{key}_results.json")
    with open(out_path, "w") as fh:
        json.dump(fold_results, fh, indent=2)
    print(f"  ✓ {out_path}")


def load_unimodal_json() -> dict:
    with open(UNIMODAL_JSON) as fh:
        return json.load(fh)

# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------
def aggregate_cm(fold_results: list) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for r in fold_results:
        cm += np.array(r["confusion_matrix"])
    return cm


def row_normalise(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def mean_per_class_f1(fold_results: list):
    arr = np.array([r["per_class_f1"] for r in fold_results])
    return arr.mean(axis=0), arr.std(axis=0)


def mean_unimodal_per_class(unimodal_data: dict, model_name: str) -> np.ndarray:
    arr = np.array([r["per_class_f1"] for r in unimodal_data[model_name]])
    return arr.mean(axis=0)


def best_unimodal_per_class(unimodal_data: dict) -> np.ndarray:
    arrays = [mean_unimodal_per_class(unimodal_data, k) for k in unimodal_data]
    return np.max(np.stack(arrays), axis=0)

# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------
def _participants_label(test_participants: list) -> str:
    return f"P{min(test_participants)}-P{max(test_participants)}"


def save_tables(
    fold_results: list,
    tables_dir: str,
    method_name: str,
    unimodal_data: dict,
):
    f1s      = [r["macro_f1"] for r in fold_results]
    accs     = [r["accuracy"]  for r in fold_results]
    mean_f1  = float(np.mean(f1s))
    std_f1   = float(np.std(f1s))
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))

    uni_means = {k: np.mean([r["macro_f1"] for r in v]) for k, v in unimodal_data.items()}
    ast_mean  = uni_means.get("AST", 0.0)
    best_uni  = max(uni_means.values())
    lift_ast  = mean_f1 - ast_mean
    lift_best = mean_f1 - best_uni

    # summary.csv
    summary_path = os.path.join(tables_dir, "summary.csv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "mean_f1", "std_f1", "mean_acc", "std_acc",
                    "n_folds", "lift_over_ast", "lift_over_best_unimodal"])
        w.writerow([method_name,
                    f"{mean_f1:.4f}", f"{std_f1:.4f}",
                    f"{mean_acc:.4f}", f"{std_acc:.4f}",
                    len(fold_results),
                    f"{lift_ast:.4f}", f"{lift_best:.4f}"])
    print(f"  ✓ {summary_path}")

    # per_fold.csv
    fold_path = os.path.join(tables_dir, "per_fold.csv")
    with open(fold_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "test_participants", "n_windows",
                    "macro_f1", "accuracy", "inference_time_s"])
        for r in fold_results:
            w.writerow([
                r["fold"],
                _participants_label(r["test_participants"]),
                r["n_windows"],
                f"{r['macro_f1']:.4f}",
                f"{r['accuracy']:.4f}",
                r["inference_time_s"],
            ])
    print(f"  ✓ {fold_path}")

    # per_class_f1.csv
    pc_means, pc_stds = mean_per_class_f1(fold_results)
    pc_path = os.path.join(tables_dir, "per_class_f1.csv")
    with open(pc_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "f1_mean", "f1_std",
                    "f1_fold0", "f1_fold1", "f1_fold2"])
        for cls_id in range(NUM_CLASSES):
            row = [
                cls_id,
                SHORT_NAMES[cls_id],
                f"{pc_means[cls_id]:.4f}",
                f"{pc_stds[cls_id]:.4f}",
            ]
            for r in fold_results:
                row.append(f"{r['per_class_f1'][cls_id]:.4f}")
            w.writerow(row)
    print(f"  ✓ {pc_path}")

    # most_confused.csv
    cm_agg = aggregate_cm(fold_results)
    np.fill_diagonal(cm_agg, 0)
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm_agg[i, j] > 0:
                pairs.append((i, j, int(cm_agg[i, j])))
    pairs.sort(key=lambda x: -x[2])
    mc_path = os.path.join(tables_dir, "most_confused.csv")
    with open(mc_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "true_class", "pred_class", "count"])
        for rank, (tc, pc, cnt) in enumerate(pairs[:10], 1):
            w.writerow([rank, SHORT_NAMES[tc], SHORT_NAMES[pc], cnt])
    print(f"  ✓ {mc_path}")

# ===========================================================================
# Figure utilities
# ===========================================================================
def _save(fig: plt.Figure, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {path}")

# ===========================================================================
# Per-method figures
# ===========================================================================

def fig_training_curves(log_data: dict, method_dir: str, cfg: dict):
    """Two subplots: loss (left) and F1 (right). One line per fold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss, ax_f1 = axes

    any_data = False
    for fold_i, data in log_data.items():
        if not data:
            continue
        any_data = True
        color  = FOLD_COLORS[fold_i]
        label  = f"Fold {fold_i} ({FOLD_TEST_PARTICIPANTS[fold_i]})"
        epochs = data["epochs"]

        ax_loss.plot(epochs, data["losses"], color=color, label=label, lw=1.5)
        ax_f1.plot(epochs,   data["f1s"],   color=color, label=label, lw=1.5)

        # Mark best F1 epoch with a star
        best_idx  = int(np.argmax(data["f1s"]))
        best_ep   = epochs[best_idx]
        ax_loss.plot(best_ep, data["losses"][best_idx], "*", color=color, ms=11, zorder=5)
        ax_f1.plot(best_ep,   data["f1s"][best_idx],   "*", color=color, ms=11, zorder=5)

    ax_f1.axhline(1 / NUM_CLASSES, ls="--", color="grey", lw=1.2, label="Random chance")

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss"); ax_loss.legend(fontsize=9)

    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Macro F1")
    ax_f1.set_title("Validation Macro F1"); ax_f1.legend(fontsize=9)

    if not any_data:
        for ax in axes:
            ax.text(0.5, 0.5, "No log data found", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=12)

    sns.despine()
    fig.suptitle(f"{cfg['name']} — Training Curves", y=1.01,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "training_curves.png"))


def fig_per_fold(fold_results: list, method_dir: str, cfg: dict):
    """Grouped bars: Macro F1 and Accuracy side by side per fold."""
    n     = len(fold_results)
    x     = np.arange(n)
    width = 0.35

    f1s     = [r["macro_f1"] for r in fold_results]
    accs    = [r["accuracy"]  for r in fold_results]
    mean_f1 = float(np.mean(f1s))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_f1  = ax.bar(x - width / 2, f1s,  width, color=cfg["color"], alpha=0.85, label="Macro F1")
    bars_acc = ax.bar(x + width / 2, accs, width, color="#FF9800",     alpha=0.85, label="Accuracy")

    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(mean_f1, ls="--", color=cfg["color"], lw=1.5,
               label=f"Mean F1 = {mean_f1:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Fold {r['fold']}\n({FOLD_TEST_PARTICIPANTS[r['fold']]})"
        for r in fold_results
    ])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"{cfg['name']} — Per-Fold Results", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "per_fold_f1.png"))


def fig_confusion_matrix_agg(fold_results: list, method_dir: str, cfg: dict):
    """Aggregated confusion matrix: sum all folds, then row-normalise."""
    cm_normed = row_normalise(aggregate_cm(fold_results))

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm_normed, ax=ax, cmap="Purples",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        linewidths=0.3, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(f"{cfg['name']} — Confusion Matrix (Aggregated, Row-Normalised)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "confusion_matrix.png"))


def fig_per_class_f1_agg(fold_results: list, method_dir: str, cfg: dict):
    """Mean ± std per-class F1. Value labels rotated 90 degrees."""
    means, stds = mean_per_class_f1(fold_results)
    mean_f1     = float(np.mean([r["macro_f1"] for r in fold_results]))
    x           = np.arange(NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=cfg["color"], alpha=0.85,
                  error_kw={"ecolor": "black", "lw": 1.2})

    for bar_obj, val, err in zip(bars, means, stds):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            val + err + 0.015,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7, rotation=90,
        )

    ax.axhline(1 / NUM_CLASSES, ls="--", color="grey",      lw=1.2, label="Random chance")
    ax.axhline(mean_f1,         ls=":",  color=cfg["color"], lw=1.5, label=f"Mean F1 = {mean_f1:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.18)
    ax.set_title(f"{cfg['name']} — Per-Class F1 (Mean ± Std)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "per_class_f1.png"))


def fig_fusion_vs_unimodal(
    fold_results: list,
    unimodal_data: dict,
    method_dir: str,
    cfg: dict,
):
    """4-bar chart: TSMixer, TinyHAR, AST, this method — F1 with error bars."""
    ordered = [
        ("TSMixer",   UNIMODAL_COLORS["TSMixer"], unimodal_data.get("TSMixer", [])),
        ("TinyHAR",   UNIMODAL_COLORS["TinyHAR"], unimodal_data.get("TinyHAR", [])),
        ("AST",       UNIMODAL_COLORS["AST"],     unimodal_data.get("AST",     [])),
        (cfg["name"], cfg["color"],               fold_results),
    ]

    names, colors, means_f1, stds_f1, means_acc = [], [], [], [], []
    for name, color, recs in ordered:
        f1s  = [r["macro_f1"] for r in recs]
        accs = [r["accuracy"]  for r in recs]
        names.append(name); colors.append(color)
        means_f1.append(float(np.mean(f1s))); stds_f1.append(float(np.std(f1s)))
        means_acc.append(float(np.mean(accs)))

    x   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means_f1, yerr=stds_f1, capsize=5, color=colors, alpha=0.85,
                  error_kw={"ecolor": "black", "lw": 1.4})

    for bar_obj, f1, acc in zip(bars, means_f1, means_acc):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 0.025,
            f"F1={f1:.4f}\nAcc={acc:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.18)
    ax.set_title(f"{cfg['name']} vs Unimodal Baselines", fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "fusion_vs_unimodal.png"))


def fig_per_class_comparison(
    fold_results: list,
    unimodal_data: dict,
    method_dir: str,
    cfg: dict,
):
    """3-group bars per class: TinyHAR, AST, this method."""
    tinyhar_means = mean_unimodal_per_class(unimodal_data, "TinyHAR")
    ast_means     = mean_unimodal_per_class(unimodal_data, "AST")
    this_means, _ = mean_per_class_f1(fold_results)

    x     = np.arange(NUM_CLASSES)
    width = 0.27

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width, tinyhar_means, width, color=UNIMODAL_COLORS["TinyHAR"],
           alpha=0.8, label="TinyHAR")
    ax.bar(x,         ast_means,     width, color=UNIMODAL_COLORS["AST"],
           alpha=0.8, label="AST")
    this_bars = ax.bar(x + width, this_means, width, color=cfg["color"],
                       alpha=0.85, label=cfg["name"])

    for bar_obj in this_bars:
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 0.01,
            f"{bar_obj.get_height():.2f}",
            ha="center", va="bottom", fontsize=6, rotation=90,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.20)
    ax.set_title(f"Per-Class F1 — TinyHAR vs AST vs {cfg['name']}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "per_class_comparison.png"))


def fig_lift_over_unimodal(
    fold_results: list,
    unimodal_data: dict,
    method_dir: str,
    cfg: dict,
):
    """Per-class delta: this method − best unimodal. Green/red coloring."""
    this_means, _ = mean_per_class_f1(fold_results)
    best_uni      = best_unimodal_per_class(unimodal_data)
    delta         = this_means - best_uni
    colors        = ["#43A047" if d >= 0 else "#E53935" for d in delta]
    x             = np.arange(NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, delta, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=1.0)

    for xi, val in zip(x, delta):
        va = "bottom" if val >= 0 else "top"
        ax.text(xi, val + (0.005 if val >= 0 else -0.005),
                f"{val:+.2f}", ha="center", va=va, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Lift vs Best Unimodal")
    ax.set_title(f"{cfg['name']} — Lift Over Best Unimodal per Class",
                 fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(method_dir, "lift_over_unimodal.png"))

# ===========================================================================
# Per-fold (single fold) figures
# ===========================================================================

def fig_confusion_matrix_single(fold_r: dict, fold_dir: str, cfg: dict):
    """Single-fold confusion matrix, row-normalised."""
    cm        = np.array(fold_r["confusion_matrix"], dtype=float)
    cm_normed = row_normalise(cm)
    fold_i    = fold_r["fold"]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm_normed, ax=ax, cmap="Purples",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        linewidths=0.3, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(
        f"{cfg['name']} — Fold {fold_i} ({FOLD_TEST_PARTICIPANTS[fold_i]}) "
        f"Confusion Matrix",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(fold_dir, "confusion_matrix.png"))


def fig_per_class_f1_single(fold_r: dict, fold_dir: str, cfg: dict):
    """Single fold per-class F1 bar chart (no error bars)."""
    f1s    = fold_r["per_class_f1"]
    fold_i = fold_r["fold"]
    x      = np.arange(NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, f1s, color=cfg["color"], alpha=0.85)

    for bar_obj in bars:
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 0.01,
            f"{bar_obj.get_height():.2f}",
            ha="center", va="bottom", fontsize=7, rotation=90,
        )

    ax.axhline(1 / NUM_CLASSES, ls="--", color="grey",      lw=1.2, label="Random chance")
    ax.axhline(fold_r["macro_f1"], ls=":", color=cfg["color"], lw=1.5,
               label=f"Macro F1 = {fold_r['macro_f1']:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.18)
    ax.set_title(
        f"{cfg['name']} — Fold {fold_i} ({FOLD_TEST_PARTICIPANTS[fold_i]}) Per-Class F1",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(fold_dir, "per_class_f1.png"))

# ===========================================================================
# Cross-method (top-level) figures
# ===========================================================================

def fig_method_comparison(
    all_results: dict,
    unimodal_data: dict,
    figures_base: str,
):
    """Horizontal bar chart: all fusion + unimodal methods, sorted by F1."""
    entries = []

    for key, fold_results in all_results.items():
        cfg = METHODS[key]
        f1s = [r["macro_f1"] for r in fold_results]
        entries.append({
            "name": cfg["name"], "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)), "color": cfg["color"],
            "alpha": 0.90, "hatch": None,
        })

    uni_color_map = {k: UNIMODAL_COLORS.get(k, "#9E9E9E") for k in unimodal_data}
    for uni_name, folds in unimodal_data.items():
        f1s = [r["macro_f1"] for r in folds]
        entries.append({
            "name": uni_name, "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)), "color": uni_color_map[uni_name],
            "alpha": 0.55, "hatch": "///",
        })

    entries.sort(key=lambda e: e["mean_f1"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, e in enumerate(entries):
        ax.barh(i, e["mean_f1"], xerr=e["std_f1"],
                color=e["color"], alpha=e["alpha"], hatch=e["hatch"],
                capsize=4, error_kw={"ecolor": "black", "lw": 1.2})
        ax.text(e["mean_f1"] + e["std_f1"] + 0.007, i,
                f"{e['mean_f1']:.4f}", va="center", fontsize=9)

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e["name"] for e in entries], fontsize=10)
    ax.set_xlabel("Mean Macro F1 (3-fold CV)")
    ax.set_xlim(0, 1.08)
    ax.set_title("Fusion vs Unimodal — All Methods", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#777", alpha=0.90,          label="Fusion"),
        Patch(facecolor="#777", alpha=0.55, hatch="///", label="Unimodal"),
    ], fontsize=9, loc="lower right")
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(figures_base, "fusion_method_comparison.png"))


def fig_per_fold_all_methods(all_results: dict, figures_base: str):
    """4 fold groups, each group has bars for every evaluated method."""
    n_methods = len(all_results)
    n_folds   = 3
    keys      = list(all_results.keys())
    x         = np.arange(n_folds)
    width     = 0.75 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, key in enumerate(keys):
        cfg          = METHODS[key]
        fold_results = all_results[key]
        f1s          = [r["macro_f1"] for r in fold_results]
        offset       = (idx - n_methods / 2 + 0.5) * width
        bars         = ax.bar(x + offset, f1s, width, color=cfg["color"],
                              alpha=0.85, label=cfg["name"])
        for bar_obj in bars:
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height() + 0.006,
                f"{bar_obj.get_height():.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Fold {i}\n({FOLD_TEST_PARTICIPANTS[i]})"
        for i in range(n_folds)
    ])
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.10)
    ax.set_title("Per-Fold F1 — All Fusion Methods", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(figures_base, "fusion_per_fold_all_methods.png"))


def fig_per_class_all_methods(
    all_results: dict,
    unimodal_data: dict,
    figures_base: str,
):
    """Line plot: one line per fusion method + AST baseline (dashed)."""
    x = np.arange(NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(16, 5))
    for key, fold_results in all_results.items():
        cfg         = METHODS[key]
        means, _    = mean_per_class_f1(fold_results)
        ax.plot(x, means, color=cfg["color"], lw=1.8, marker="o", ms=5, label=cfg["name"])

    ast_means = mean_unimodal_per_class(unimodal_data, "AST")
    ax.plot(x, ast_means, color=UNIMODAL_COLORS["AST"], lw=1.4, ls="--",
            marker="s", ms=4, alpha=0.75, label="AST (baseline)")

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 — All Fusion Methods", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(figures_base, "fusion_per_class_all_methods.png"))


def fig_leaderboard(
    all_results: dict,
    unimodal_data: dict,
    figures_base: str,
):
    """Lollipop chart sorted by mean F1. Vertical dashed reference lines for unimodals."""
    entries = []
    for key, fold_results in all_results.items():
        cfg = METHODS[key]
        f1s = [r["macro_f1"] for r in fold_results]
        entries.append({
            "name": cfg["name"], "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)), "color": cfg["color"],
        })
    entries.sort(key=lambda e: e["mean_f1"])

    uni_refs = {
        k: float(np.mean([r["macro_f1"] for r in v]))
        for k, v in unimodal_data.items()
    }
    uni_ls = {"TinyHAR": ":", "AST": "--", "TSMixer": "-."}

    fig, ax = plt.subplots(figsize=(9, max(4, len(entries) * 1.2)))
    y_pos = np.arange(len(entries))

    for i, e in enumerate(entries):
        ax.hlines(i, 0, e["mean_f1"], color=e["color"], lw=2.5, alpha=0.7)
        ax.plot(e["mean_f1"], i, "o", color=e["color"], ms=11)
        ax.errorbar(e["mean_f1"], i, xerr=e["std_f1"],
                    fmt="none", ecolor=e["color"], capsize=5, elinewidth=1.8)
        ax.text(
            e["mean_f1"] + e["std_f1"] + 0.007, i,
            f"{e['mean_f1']:.4f} ± {e['std_f1']:.4f}",
            va="center", fontsize=9,
        )

    for uni_name, ref_val in uni_refs.items():
        ax.axvline(ref_val,
                   ls=uni_ls.get(uni_name, "--"),
                   color=UNIMODAL_COLORS.get(uni_name, "grey"),
                   lw=1.3, alpha=0.75, label=f"{uni_name} ({ref_val:.4f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["name"] for e in entries], fontsize=10)
    ax.set_xlabel("Mean Macro F1 (3-fold CV)")
    ax.set_title("Fusion Method Leaderboard", fontsize=13, fontweight="bold")
    max_val = max(e["mean_f1"] + e["std_f1"] for e in entries) if entries else 1.0
    ax.set_xlim(0, min(max_val + 0.14, 1.05))
    ax.legend(fontsize=8, loc="lower right")
    sns.despine()
    fig.tight_layout()
    _save(fig, os.path.join(figures_base, "fusion_leaderboard.png"))

# ===========================================================================
# Summary printout
# ===========================================================================
def print_summary(all_results: dict, unimodal_data: dict):
    uni_means    = {k: float(np.mean([r["macro_f1"] for r in v])) for k, v in unimodal_data.items()}
    best_uni     = max(uni_means.values())
    best_uni_name = max(uni_means, key=uni_means.get)
    ast_mean     = uni_means.get("AST", 0.0)

    print("\n" + "=" * 72)
    print("FUSION EVALUATION SUMMARY")
    print("=" * 72)
    print(f"{'Method':<18} {'Mean F1':>10} {'Std F1':>8} {'Acc':>8}"
          f" {'vs AST':>8} {'vs Best Uni':>12}")
    print("-" * 72)
    for key, fold_results in sorted(
        all_results.items(), key=lambda kv: -np.mean([r["macro_f1"] for r in kv[1]])
    ):
        f1s  = [r["macro_f1"] for r in fold_results]
        accs = [r["accuracy"]  for r in fold_results]
        m, s, a = float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(accs))
        print(f"{METHODS[key]['name']:<18} {m:>10.4f} {s:>8.4f} {a:>8.4f}"
              f" {m - ast_mean:>+8.4f} {m - best_uni:>+12.4f}")
    print("-" * 72)
    print(f"Best unimodal: {best_uni_name} (F1={best_uni:.4f})")
    print("Unimodal: " +
          " | ".join(f"{k}={v:.4f}" for k, v in sorted(uni_means.items(), key=lambda x: -x[1])))
    print()

# ===========================================================================
# Main
# ===========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"Models dir:  {MODELS_DIR}")
    print(f"Figures dir: {FIGURES_BASE}")
    print(f"Results dir: {RESULTS_BASE}")

    os.makedirs(FIGURES_BASE, exist_ok=True)
    os.makedirs(RESULTS_BASE, exist_ok=True)

    # Load unimodal baselines
    if not os.path.exists(UNIMODAL_JSON):
        print(f"ERROR: unimodal results not found at {UNIMODAL_JSON}")
        sys.exit(1)
    unimodal_data = load_unimodal_json()
    print(f"Loaded unimodal data: {list(unimodal_data.keys())}")

    # CV splits
    participants = list(range(1, 21))
    splits       = get_3fold_splits(participants)

    all_results: dict = {}

    # ------------------------------------------------------------------
    # Per-method evaluation loop
    # ------------------------------------------------------------------
    for key, cfg in METHODS.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {cfg['name']} ...")

        method_dir = os.path.join(FIGURES_BASE, key)
        tables_dir = os.path.join(method_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        fold_results = run_inference_all_folds(key, cfg, splits, device)

        if not fold_results:
            print(f"  No checkpoints found for {cfg['name']}, skipping.")
            continue

        all_results[key] = fold_results

        save_json(fold_results, key)
        save_tables(fold_results, tables_dir, cfg["name"], unimodal_data)

        log_data = {i: parse_log(cfg["ckpt"], i) for i in range(3)}

        fig_training_curves(log_data, method_dir, cfg)
        fig_per_fold(fold_results, method_dir, cfg)
        fig_confusion_matrix_agg(fold_results, method_dir, cfg)
        fig_per_class_f1_agg(fold_results, method_dir, cfg)
        fig_fusion_vs_unimodal(fold_results, unimodal_data, method_dir, cfg)
        fig_per_class_comparison(fold_results, unimodal_data, method_dir, cfg)
        fig_lift_over_unimodal(fold_results, unimodal_data, method_dir, cfg)

        for fold_r in fold_results:
            fold_dir = os.path.join(method_dir, f"fold{fold_r['fold']}")
            os.makedirs(fold_dir, exist_ok=True)
            fig_confusion_matrix_single(fold_r, fold_dir, cfg)
            fig_per_class_f1_single(fold_r, fold_dir, cfg)

    # ------------------------------------------------------------------
    # Cross-method figures
    # ------------------------------------------------------------------
    if len(all_results) >= 2:
        print(f"\n{'=' * 60}")
        print("Generating cross-method figures ...")
        fig_method_comparison(all_results, unimodal_data, FIGURES_BASE)
        fig_per_fold_all_methods(all_results, FIGURES_BASE)
        fig_per_class_all_methods(all_results, unimodal_data, FIGURES_BASE)
        fig_leaderboard(all_results, unimodal_data, FIGURES_BASE)
    elif len(all_results) == 1:
        print("\nOnly 1 method evaluated — skipping cross-method figures.")
    else:
        print("\nNo methods evaluated — check checkpoint paths in MODELS_DIR.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_results:
        print_summary(all_results, unimodal_data)

    print("Done.")


if __name__ == "__main__":
    main()
