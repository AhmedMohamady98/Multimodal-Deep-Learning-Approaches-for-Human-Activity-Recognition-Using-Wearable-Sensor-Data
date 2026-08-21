"""
Unimodal 10s evaluation script.

Mirrors evaluate_unimodal_8s.py exactly but for 10-second window models.
Loads checkpoints from results/v3/10s/models/, outputs to results/v3/10s/.

Output structure:
    results/v3/10s/figures/
    ├── tinyhar/
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── per_class_f1.png
    │   ├── fold0/ {confusion_matrix, per_class_f1}.png
    │   ├── fold1/ ...
    │   └── fold2/ ...
    ├── tsmixer/  (same structure)
    ├── ast/      (same structure)
    ├── summary_comparison.png
    ├── modality_comparison.png
    ├── accuracy_vs_f1.png
    ├── class_distribution.png
    └── convergence_comparison.png

    results/v3/10s/tables/
    ├── unimodal_summary.csv
    ├── per_fold_results.csv
    ├── per_class_f1.csv
    ├── per_class_precision_recall.csv
    ├── dataset_statistics.csv
    ├── training_efficiency.csv
    ├── mcnemar_tests.csv
    └── most_confused_pairs.csv

    results/v3/10s/unimodal_results.json

Usage:
    PYTHONPATH=. python jobs/evaluate_unimodal_10s.py
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
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix)
from scipy.stats import chi2_contingency

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.data.splits import get_3fold_splits
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR  = "results/v3/10s/models"
FIGURES_DIR = "results/v3/10s/figures"
TABLES_DIR  = "results/v3/10s/tables"
JSON_OUT    = "results/v3/10s/unimodal_results.json"

# 10-second window parameters
WINDOW_SIZE = 500   # 10s × 50 Hz
STEP        = 250   # 50% overlap

# Log file discovery — use glob to find latest matching logs
def _find_log(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None

LOG_FILES = {
    "TinyHAR": _find_log("logs/tinyhar_tsmixer_10s_*.out"),
    "TSMixer": _find_log("logs/tinyhar_tsmixer_10s_*.out"),  # same combined log
    "AST_fold0": _find_log("logs/ast_10s_fold0_*.out"),
    "AST_fold1": _find_log("logs/ast_10s_fold1_*.out"),
    "AST_fold2": _find_log("logs/ast_10s_fold2_*.out"),
}

MODEL_TAGS  = {"TinyHAR": "tinyhar", "TSMixer": "tsmixer", "AST": "ast"}
CLASS_NAMES = [ID_TO_ACTIVITY[i] for i in range(NUM_CLASSES)]
SHORT_NAMES = [
    "Wash hands", "Floor clean", "Window clean", "Vacuum",
    "Brush teeth", "Wash dishes", "Put away dishes", "Make tea",
    "Cut veg", "Cream hands", "Drink", "Water plants",
    "Clean table", "Disinfect", "Clean dishwasher", "Null",
]

COLORS      = {"TinyHAR": "#2196F3", "TSMixer": "#4CAF50", "AST": "#FF5722"}
FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


# ── Model definitions ─────────────────────────────────────────────────────────

class UnimodalModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


class ASTModel(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.encoder    = ASTEncoder(embed_dim=embed_dim, freeze_backbone=True)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, audio_list):
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len,
                              device=audio_list[0].device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a
        return self.classifier(self.encoder(padded))


def make_model(name):
    embed_dim = 128
    if name == "TinyHAR":
        return UnimodalModel(TinyHAREncoder(in_channels=12, embed_dim=embed_dim),
                             embed_dim, NUM_CLASSES), "imu"
    elif name == "TSMixer":
        # seq_len must match 8s window size
        return UnimodalModel(TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1,
                                            embed_dim=embed_dim),
                             embed_dim, NUM_CLASSES), "humidity"
    elif name == "AST":
        return ASTModel(embed_dim, NUM_CLASSES), "audio"


def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, modality):
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        if modality == "imu":
            logits = model(imu)
        elif modality == "humidity":
            logits = model(humidity)
        elif modality == "audio":
            audio_dev = [a.to(device) for a in audio]
            logits = model(audio_dev)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    elapsed = time.time() - t0
    return np.concatenate(all_preds), np.concatenate(all_labels), elapsed


def evaluate_model(name, splits, device):
    tag      = MODEL_TAGS[name]
    modality = {"TinyHAR": "imu", "TSMixer": "humidity", "AST": "audio"}[name]

    fold_results = []
    cm_sum       = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for fold_idx, fold in enumerate(splits):
        ckpt = os.path.join(MODELS_DIR, f"{tag}_fold{fold_idx}.pt")
        if not os.path.exists(ckpt):
            print(f"  ⚠️  Missing: {ckpt} — skipping fold {fold_idx}")
            continue

        model, _ = make_model(name)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)

        # Use 8s window size for dataset loading
        test_ds     = HARMESDataset(fold["test"], DATA_DIR,
                                    window_size=WINDOW_SIZE, step=STEP)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4)

        all_labels_list = [test_ds[i][3].item() for i in range(len(test_ds))]
        class_counts = np.bincount(all_labels_list, minlength=NUM_CLASSES)

        preds, labels, infer_time = run_inference(model, test_loader, device, modality)

        f1           = f1_score(labels, preds, average="macro", zero_division=0)
        acc          = accuracy_score(labels, preds)
        per_class_f1 = f1_score(labels, preds, average=None,
                                labels=list(range(NUM_CLASSES)), zero_division=0)
        per_class_prec = __import__("sklearn.metrics", fromlist=["precision_score"])\
                         .precision_score(labels, preds, average=None,
                                         labels=list(range(NUM_CLASSES)), zero_division=0)
        per_class_rec  = __import__("sklearn.metrics", fromlist=["recall_score"])\
                         .recall_score(labels, preds, average=None,
                                       labels=list(range(NUM_CLASSES)), zero_division=0)
        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

        cm_sum += cm
        fold_results.append({
            "fold":               fold_idx,
            "test_participants":  fold["test"],
            "train_participants": fold["train"],
            "n_windows":          len(test_ds),
            "class_counts":       class_counts.tolist(),
            "macro_f1":           float(f1),
            "accuracy":           float(acc),
            "per_class_f1":       per_class_f1.tolist(),
            "per_class_precision":per_class_prec.tolist(),
            "per_class_recall":   per_class_rec.tolist(),
            "confusion_matrix":   cm.tolist(),
            "inference_time_s":   float(infer_time),
            "preds":              preds.tolist(),
            "labels":             labels.tolist(),
        })
        print(f"  Fold {fold_idx}: F1={f1:.4f}  acc={acc:.4f}  "
              f"test={fold['test']}  n={len(test_ds):,}")

    return fold_results, cm_sum


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_training_log(log_path, model_name=None):
    """Parse training log — handles both combined and single-fold logs."""
    folds     = {}
    curr_fold = None
    fold_re   = re.compile(r"(?:TinyHAR|TSMixer)\s+[—-]\s+Fold\s+(\d+)|Fold\s+(\d+)")
    epoch_re  = re.compile(
        r"Epoch\s+(\d+)/\d+\s+loss=([\d.]+)\s+F1=([\d.]+)\s+acc=([\d.]+)"
    )
    stop_re  = re.compile(r"Early stopping at epoch\s+(\d+)")
    start_re = re.compile(r"Started:\s+(.+)")
    end_re   = re.compile(r"Finished:\s+(.+)")

    meta = {"start": None, "end": None}

    # For combined log (TinyHAR+TSMixer), filter by model section
    in_model_section = (model_name is None)

    with open(log_path) as f:
        for line in f:
            if m := start_re.search(line):
                meta["start"] = m.group(1).strip()
            if m := end_re.search(line):
                meta["end"] = m.group(1).strip()

            # Detect model section boundaries in combined log
            if model_name == "TinyHAR" and "TinyHAR (IMU)" in line:
                in_model_section = True
            elif model_name == "TSMixer" and "TSMixer (Humidity)" in line:
                in_model_section = True
            elif model_name in ("TinyHAR", "TSMixer"):
                # End section when the other model starts
                other = "TSMixer" if model_name == "TinyHAR" else "TinyHAR"
                if f"{other} (IMU)" in line or f"{other} (Humidity)" in line:
                    in_model_section = False

            if not in_model_section:
                continue

            if m := fold_re.search(line):
                if "train=" in line:
                    fold_num = int(m.group(1) or m.group(2))
                    curr_fold = fold_num
                    folds[curr_fold] = {"epochs": [], "early_stop": None}
                    continue
            if m := epoch_re.search(line):
                if curr_fold is not None:
                    folds[curr_fold]["epochs"].append({
                        "epoch": int(m.group(1)),
                        "loss":  float(m.group(2)),
                        "f1":    float(m.group(3)),
                        "acc":   float(m.group(4)),
                    })
            if m := stop_re.search(line):
                if curr_fold is not None:
                    folds[curr_fold]["early_stop"] = int(m.group(1))

    return folds, meta


def parse_ast_logs():
    """Parse AST from 3 separate fold logs and merge into one fold dict."""
    merged = {}
    meta   = {"start": None, "end": None}
    for fold_idx in range(3):
        log_path = LOG_FILES.get(f"AST_fold{fold_idx}")
        if not log_path or not os.path.exists(log_path):
            print(f"  AST fold {fold_idx}: log not found")
            continue
        fold_dict, fold_meta = parse_training_log(log_path)
        # Single-fold log has fold 0 internally — remap to correct fold index
        for internal_idx, info in fold_dict.items():
            merged[fold_idx] = info
        if fold_meta["start"]:
            meta["start"] = fold_meta["start"]
        if fold_meta["end"]:
            meta["end"] = fold_meta["end"]
    return merged, meta


# ── Dataset statistics ────────────────────────────────────────────────────────

def compute_dataset_stats(splits):
    stats = []
    for fold_idx, fold in enumerate(splits):
        for split_name, participant_ids in [("train", fold["train"]),
                                             ("test",  fold["test"])]:
            ds = HARMESDataset(participant_ids, DATA_DIR,
                               window_size=WINDOW_SIZE, step=STEP)
            all_labels = [ds[i][3].item() for i in range(len(ds))]
            counts = np.bincount(all_labels, minlength=NUM_CLASSES)
            for class_id, count in enumerate(counts):
                stats.append({
                    "fold":       fold_idx,
                    "split":      split_name,
                    "class_id":   class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "n_windows":  int(count),
                })
    return stats


# ── Statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(preds_a, preds_b, labels):
    correct_a = (np.array(preds_a) == np.array(labels))
    correct_b = (np.array(preds_b) == np.array(labels))
    n01 = np.sum(~correct_a &  correct_b)
    n10 = np.sum( correct_a & ~correct_b)
    if (n01 + n10) == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return float(chi2), float(p)


def compute_most_confused_pairs(cm, top_n=5):
    cm_copy = cm.copy().astype(float)
    np.fill_diagonal(cm_copy, 0)
    flat    = cm_copy.flatten()
    indices = np.argsort(flat)[::-1][:top_n]
    pairs   = []
    for idx in indices:
        true_cls = idx // NUM_CLASSES
        pred_cls = idx  % NUM_CLASSES
        pairs.append({
            "true_class": CLASS_NAMES[true_cls],
            "pred_class": CLASS_NAMES[pred_cls],
            "count":      int(cm_copy[true_cls, pred_cls]),
        })
    return pairs


# ── Table writers ─────────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ {path}")


def save_tables(results_data, cm_data, log_data, dataset_stats):
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 1. Summary
    rows = []
    for name, folds in results_data.items():
        if not folds: continue
        f1s  = [r["macro_f1"] for r in folds]
        accs = [r["accuracy"]  for r in folds]
        rows.append({
            "model":    name,
            "mean_f1":  f"{np.mean(f1s):.4f}",
            "std_f1":   f"{np.std(f1s):.4f}",
            "mean_acc": f"{np.mean(accs):.4f}",
            "std_acc":  f"{np.std(accs):.4f}",
            "n_folds":  len(folds),
        })
    write_csv(os.path.join(TABLES_DIR, "unimodal_summary.csv"),
              rows, ["model", "mean_f1", "std_f1", "mean_acc", "std_acc", "n_folds"])

    # 2. Per-fold
    rows = []
    for name, folds in results_data.items():
        for r in folds:
            rows.append({
                "model":             name,
                "fold":              r["fold"],
                "test_participants": str(r["test_participants"]),
                "n_windows":         r["n_windows"],
                "macro_f1":          f"{r['macro_f1']:.4f}",
                "accuracy":          f"{r['accuracy']:.4f}",
                "inference_time_s":  f"{r['inference_time_s']:.1f}",
            })
    write_csv(os.path.join(TABLES_DIR, "per_fold_results.csv"),
              rows, ["model", "fold", "test_participants", "n_windows",
                     "macro_f1", "accuracy", "inference_time_s"])

    # 3. Per-class F1
    rows = []
    for class_id in range(NUM_CLASSES):
        row = {"class_id": class_id, "class_name": CLASS_NAMES[class_id],
               "short_name": SHORT_NAMES[class_id]}
        for name, folds in results_data.items():
            if not folds:
                row[f"{name}_f1_mean"] = ""
                row[f"{name}_f1_std"]  = ""
                continue
            vals = [r["per_class_f1"][class_id] for r in folds]
            row[f"{name}_f1_mean"] = f"{np.mean(vals):.4f}"
            row[f"{name}_f1_std"]  = f"{np.std(vals):.4f}"
        rows.append(row)
    fields = ["class_id", "class_name", "short_name"]
    for name in results_data:
        fields += [f"{name}_f1_mean", f"{name}_f1_std"]
    write_csv(os.path.join(TABLES_DIR, "per_class_f1.csv"), rows, fields)

    # 4. Per-class precision & recall
    rows = []
    for class_id in range(NUM_CLASSES):
        row = {"class_id": class_id, "class_name": CLASS_NAMES[class_id]}
        for name, folds in results_data.items():
            if not folds: continue
            prec = np.mean([r["per_class_precision"][class_id] for r in folds])
            rec  = np.mean([r["per_class_recall"][class_id]    for r in folds])
            f1   = np.mean([r["per_class_f1"][class_id]        for r in folds])
            row[f"{name}_precision"] = f"{prec:.4f}"
            row[f"{name}_recall"]    = f"{rec:.4f}"
            row[f"{name}_f1"]        = f"{f1:.4f}"
        rows.append(row)
    fields = ["class_id", "class_name"]
    for name in results_data:
        fields += [f"{name}_precision", f"{name}_recall", f"{name}_f1"]
    write_csv(os.path.join(TABLES_DIR, "per_class_precision_recall.csv"), rows, fields)

    # 5. Dataset statistics
    write_csv(os.path.join(TABLES_DIR, "dataset_statistics.csv"),
              dataset_stats,
              ["fold", "split", "class_id", "class_name", "n_windows"])

    # 6. Training efficiency
    rows = []
    for name, fold_dict in log_data.items():
        for fold_idx, fold_info in fold_dict.items():
            epochs     = fold_info["epochs"]
            early_stop = fold_info.get("early_stop")
            best_epoch = max(epochs, key=lambda e: e["f1"])["epoch"] if epochs else None
            best_f1    = max(e["f1"] for e in epochs) if epochs else None
            rows.append({
                "model":            name,
                "fold":             fold_idx,
                "total_epochs":     len(epochs),
                "best_epoch":       best_epoch,
                "best_f1":          f"{best_f1:.4f}" if best_f1 else "",
                "early_stop_epoch": early_stop if early_stop else "no",
                "converged_early":  "yes" if early_stop else "no",
            })
    write_csv(os.path.join(TABLES_DIR, "training_efficiency.csv"),
              rows, ["model", "fold", "total_epochs", "best_epoch",
                     "best_f1", "early_stop_epoch", "converged_early"])

    # 7. McNemar's test
    model_names  = list(results_data.keys())
    mcnemar_rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            folds_a, folds_b = results_data[name_a], results_data[name_b]
            for fold_idx in range(min(len(folds_a), len(folds_b))):
                ra = next((r for r in folds_a if r["fold"] == fold_idx), None)
                rb = next((r for r in folds_b if r["fold"] == fold_idx), None)
                if ra is None or rb is None: continue
                chi2, p = mcnemar_test(ra["preds"], rb["preds"], ra["labels"])
                mcnemar_rows.append({
                    "model_a": name_a, "model_b": name_b, "fold": fold_idx,
                    "chi2": f"{chi2:.4f}", "p_value": f"{p:.4f}",
                    "significant": "yes" if p < 0.05 else "no",
                })
    write_csv(os.path.join(TABLES_DIR, "mcnemar_tests.csv"),
              mcnemar_rows,
              ["model_a", "model_b", "fold", "chi2", "p_value", "significant"])

    # 8. Most confused pairs
    confused_rows = []
    for name, cm in cm_data.items():
        pairs = compute_most_confused_pairs(cm, top_n=10)
        for rank, pair in enumerate(pairs, 1):
            confused_rows.append({"model": name, "rank": rank, **pair})
    write_csv(os.path.join(TABLES_DIR, "most_confused_pairs.csv"),
              confused_rows,
              ["model", "rank", "true_class", "pred_class", "count"])


# ── Figures ───────────────────────────────────────────────────────────────────

def _save(fig, directory, filename):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def fig_training_curves(fold_dict, name, out_dir):
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{name} (8s) — Training Curves", fontweight="bold")
    for fold_idx, fold_info in fold_dict.items():
        epochs = fold_info["epochs"]
        if not epochs: continue
        xs = [e["epoch"] for e in epochs]
        c  = FOLD_COLORS[fold_idx % len(FOLD_COLORS)]
        ax_loss.plot(xs, [e["loss"] for e in epochs],
                     label=f"Fold {fold_idx}", color=c, linewidth=2, alpha=0.9)
        ax_f1.plot(xs, [e["f1"] for e in epochs],
                   label=f"Fold {fold_idx}", color=c, linewidth=2, alpha=0.9)
        best = max(epochs, key=lambda e: e["f1"])
        ax_f1.scatter([best["epoch"]], [best["f1"]], color=c, s=80, zorder=5, marker="*")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Training Loss"); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Macro F1")
    ax_f1.set_title("Test Macro F1"); ax_f1.set_ylim(0, 1)
    ax_f1.axhline(1/NUM_CLASSES, color="gray", linestyle="--",
                  linewidth=1, label="Random chance")
    ax_f1.legend(); ax_f1.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "training_curves.png")


def fig_confusion_matrix(cm, name, out_dir, title_suffix=""):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_norm, ax=ax,
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                cmap="Blues", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor="lightgray",
                cbar_kws={"shrink": 0.8, "label": "Recall"})
    ax.set_title(f"{name} (8s) — Confusion Matrix{title_suffix}", fontweight="bold")
    ax.set_xlabel("Predicted Activity"); ax.set_ylabel("True Activity")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    _save(fig, out_dir, "confusion_matrix.png")


def fig_per_class_f1(means, stds, name, out_dir, title_suffix="", color=None):
    color = color or COLORS.get(name, "#607D8B")
    xs    = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(13, 5))
    if stds is not None:
        ax.bar(xs, means, color=color, alpha=0.85,
               yerr=stds, capsize=4, error_kw={"linewidth": 1.5})
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.04, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    else:
        ax.bar(xs, means, color=color, alpha=0.85)
        for x, m in zip(xs, means):
            ax.text(x, m + 0.04, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--",
               linewidth=1.2, label=f"Random ({1/NUM_CLASSES:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.25)
    ax.set_title(f"{name} (8s) — Per-Class F1{title_suffix}", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "per_class_f1.png")


def fig_summary_comparison(results_data, out_dir):
    names = list(results_data.keys())
    means, stds, accs = [], [], []
    for name, folds in results_data.items():
        f1s = [r["macro_f1"] for r in folds]
        ac  = [r["accuracy"]  for r in folds]
        means.append(np.mean(f1s)); stds.append(np.std(f1s)); accs.append(np.mean(ac))
    fig, ax = plt.subplots(figsize=(8, 5))
    xs   = np.arange(len(names))
    bars = ax.bar(xs, means, 0.5, yerr=stds, capsize=7,
                  color=[COLORS[n] for n in names], alpha=0.88,
                  error_kw={"linewidth": 2})
    for bar, m, s, a in zip(bars, means, stds, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.015,
                f"F1={m:.3f}\nAcc={a:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--", linewidth=1.2,
               label=f"Random chance ({1/NUM_CLASSES:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Macro F1 Score"); ax.set_ylim(0, 1)
    ax.set_title("Unimodal Results (10s Windows) — HARMES (20 Participants, 3-Fold CV)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "summary_comparison.png")


def fig_modality_comparison(results_data, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle("Per-Class F1 — All Modalities Compared (10s Windows)",
                 fontweight="bold", fontsize=13)
    xs = np.arange(NUM_CLASSES)
    for ax, (name, folds) in zip(axes, results_data.items()):
        if not folds: continue
        per_class = np.array([r["per_class_f1"] for r in folds])
        means = per_class.mean(axis=0); stds = per_class.std(axis=0)
        ax.bar(xs, means, color=COLORS[name], alpha=0.85,
               yerr=stds, capsize=3, error_kw={"linewidth": 1.2})
        ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, max(means + stds) * 1.35 + 0.05)
        ax.set_title(f"{name} (10s)", fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "modality_comparison.png")


def fig_accuracy_vs_f1(results_data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, folds in results_data.items():
        f1s  = [r["macro_f1"] for r in folds]
        accs = [r["accuracy"]  for r in folds]
        ax.scatter(f1s, accs, color=COLORS[name], s=120, label=name,
                   zorder=3, edgecolors="white", linewidths=1)
        for i, (f, a) in enumerate(zip(f1s, accs)):
            ax.annotate(f"F{i}", (f, a), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=COLORS[name])
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="F1 = Accuracy")
    ax.set_xlabel("Macro F1 Score"); ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs Macro F1 — 10s Windows", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "accuracy_vs_f1.png")


def fig_class_distribution(dataset_stats, out_dir):
    fold0 = [s for s in dataset_stats if s["fold"] == 0]
    train = {s["class_name"]: s["n_windows"] for s in fold0 if s["split"] == "train"}
    test  = {s["class_name"]: s["n_windows"] for s in fold0 if s["split"] == "test"}
    xs    = np.arange(NUM_CLASSES)
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(xs - width/2, [train.get(c, 0) for c in CLASS_NAMES],
           width, label="Train", color="#2196F3", alpha=0.85)
    ax.bar(xs + width/2, [test.get(c,  0) for c in CLASS_NAMES],
           width, label="Test",  color="#FF5722", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Windows")
    ax.set_title("Class Distribution (10s Windows) — Fold 0 (Train vs Test)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "class_distribution.png")


def fig_convergence_comparison(log_data, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(log_data.keys())
    xs    = np.arange(3)
    width = 0.25
    for i, name in enumerate(model_names):
        best_epochs = []
        for fold_idx in range(3):
            fold_info   = log_data[name].get(fold_idx, {"epochs": []})
            epochs      = fold_info["epochs"]
            be = max(epochs, key=lambda e: e["f1"])["epoch"] if epochs else 0
            best_epochs.append(be)
        bars = ax.bar(xs + i*width, best_epochs, width, label=name,
                      color=COLORS[name], alpha=0.85)
        for bar, val in zip(bars, best_epochs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs + width)
    ax.set_xticklabels(["Fold 0", "Fold 1", "Fold 2"])
    ax.set_ylabel("Best Epoch")
    ax.set_title("Convergence Speed (10s Windows) — Best Epoch per Fold",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, "convergence_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE/50:.0f}s  |  "
          f"Step: {STEP} samples (50% overlap)\n")

    splits = get_3fold_splits(list(range(1, 21)))

    # ── 1. Run inference ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Running inference from 8s checkpoints...")
    print("=" * 60)
    results_data, cm_data = {}, {}
    for name in ["TinyHAR", "TSMixer", "AST"]:
        print(f"\n{name}:")
        fold_results, cm_sum = evaluate_model(name, splits, device)
        results_data[name]   = fold_results
        cm_data[name]        = cm_sum

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w") as f:
        clean = {}
        for name, folds in results_data.items():
            clean[name] = [{k: v for k, v in r.items()
                            if k not in ("preds", "labels")} for r in folds]
        json.dump(clean, f, indent=2)
    print(f"\n  ✓ {JSON_OUT}")

    # ── 2. Parse logs ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Parsing training logs...")
    print("=" * 60)
    log_data = {}

    for name in ["TinyHAR", "TSMixer"]:
        log_path = LOG_FILES.get(name)
        if log_path and os.path.exists(log_path):
            fold_dict, meta = parse_training_log(log_path, model_name=name)
            log_data[name]  = fold_dict
            total = sum(len(v["epochs"]) for v in fold_dict.values())
            print(f"  {name}: {total} epoch entries")
        else:
            print(f"  {name}: log not found")

    ast_folds, ast_meta = parse_ast_logs()
    if ast_folds:
        log_data["AST"] = ast_folds
        total = sum(len(v["epochs"]) for v in ast_folds.values())
        print(f"  AST: {total} epoch entries across {len(ast_folds)} folds")

    # ── 3. Dataset statistics ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Computing dataset statistics (8s windows)...")
    print("=" * 60)
    dataset_stats = compute_dataset_stats(splits)
    print(f"  Done — {len(dataset_stats)} rows")

    # ── 4. Tables ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving tables...")
    print("=" * 60)
    save_tables(results_data, cm_data, log_data, dataset_stats)

    # ── 5. Figures ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for name in ["TinyHAR", "TSMixer", "AST"]:
        tag       = MODEL_TAGS[name]
        model_dir = os.path.join(FIGURES_DIR, tag)
        folds     = results_data[name]
        print(f"\n{name}:")

        if name in log_data:
            fig_training_curves(log_data[name], name, model_dir)
        if cm_data[name].sum() > 0:
            fig_confusion_matrix(cm_data[name], name, model_dir,
                                 " (Averaged across Folds)")
        if folds:
            per_class = np.array([r["per_class_f1"] for r in folds])
            fig_per_class_f1(per_class.mean(axis=0), per_class.std(axis=0),
                             name, model_dir, " (Mean ± Std across Folds)")
        for r in folds:
            fold_idx = r["fold"]
            fold_dir = os.path.join(model_dir, f"fold{fold_idx}")
            fig_confusion_matrix(np.array(r["confusion_matrix"]), name, fold_dir,
                                 f" — Fold {fold_idx} (test={r['test_participants']})")
            fig_per_class_f1(np.array(r["per_class_f1"]), None, name, fold_dir,
                             f" — Fold {fold_idx} (test={r['test_participants']})")

    print(f"\nCross-model figures:")
    fig_summary_comparison(results_data, FIGURES_DIR)
    fig_modality_comparison(results_data, FIGURES_DIR)
    fig_accuracy_vs_f1(results_data, FIGURES_DIR)
    fig_class_distribution(dataset_stats, FIGURES_DIR)
    if log_data:
        fig_convergence_comparison(log_data, FIGURES_DIR)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("10s UNIMODAL RESULTS SUMMARY (20 participants, 3-fold CV)")
    print("=" * 60)
    print(f"{'Model':<12} {'Macro F1':>10} {'± std':>8} {'Accuracy':>10}")
    print("-" * 45)
    for name, folds in results_data.items():
        if not folds: continue
        f1s  = [r["macro_f1"] for r in folds]
        accs = [r["accuracy"]  for r in folds]
        print(f"{name:<12} {np.mean(f1s):>10.4f} {np.std(f1s):>8.4f}"
              f" {np.mean(accs):>10.4f}")
        for r in folds:
            print(f"  Fold {r['fold']}: F1={r['macro_f1']:.4f}  "
                  f"acc={r['accuracy']:.4f}  n={r['n_windows']:,} windows")

    print(f"\nFigures → {FIGURES_DIR}/")
    print(f"Tables  → {TABLES_DIR}/")
    print(f"JSON    → {JSON_OUT}")


if __name__ == "__main__":
    main()
