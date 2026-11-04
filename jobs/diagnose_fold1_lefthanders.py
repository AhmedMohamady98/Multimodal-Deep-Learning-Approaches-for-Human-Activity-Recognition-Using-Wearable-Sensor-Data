"""
Left-hander impact analysis — fold 1 per-participant breakdown.

Runs fold 1 checkpoints for ALL models (3 unimodal + 5 fusion) and evaluates
each test participant (P8–P14) individually.  Highlights P10 and P14 (identified
as likely left-handed) and answers the core research question:

    "Which fusion strategy best compensates for participant handedness?"

Key comparisons produced:
  1. Per-participant F1 table for every model
  2. Left-hander gap: mean(P10, P14) vs mean(P8,P9,P11,P12,P13) per model
  3. Per-class F1 for P10 and P14 across all models
  4. Summary ranking: which model minimises the left-hander gap

Output:
  results/lefthander/
  ├── per_participant_all_models.csv
  ├── lefthander_gap_summary.csv
  ├── p10_per_class.csv
  ├── p14_per_class.csv
  ├── lefthander_gap.png          ← main comparison figure
  ├── per_participant_heatmap.png ← all models × all participants
  ├── p10_per_class_heatmap.png
  └── p14_per_class_heatmap.png

Usage:
    PYTHONPATH=. python jobs/diagnose_fold1_lefthanders.py
    PYTHONPATH=. python jobs/diagnose_fold1_lefthanders.py --models unimodal
    PYTHONPATH=. python jobs/diagnose_fold1_lefthanders.py --models fusion
    PYTHONPATH=. python jobs/diagnose_fold1_lefthanders.py --models TinyHAR GMF MBT
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.data.dataset import HARMESDataset, NUM_CLASSES
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.late_fusion import LateFusion
from src.fusion.gmf import GatedMultimodalFusion
from src.fusion.mbt import MultimodalBottleneckTransformer
from src.fusion.cma import CrossModalAttention
from src.fusion.lmf import LowRankMultimodalFusion

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR  = "results/models"
OUT_DIR     = "results/lefthander"

WINDOW_SIZE = 500
STEP        = 250
EMBED_DIM   = 128
FOLD        = 1

FOLD1_ALL        = [8, 9, 10, 11, 12, 13, 14]
LEFT_HANDERS     = [10, 14]
RIGHT_HANDERS    = [8, 9, 11, 12, 13]

SHORT_NAMES = [
    "Wash hands", "Floor clean", "Window clean", "Vacuum",
    "Brush teeth", "Wash dishes", "Put away dishes", "Make tea",
    "Cut veg", "Cream hands", "Drink", "Water plants",
    "Clean table", "Disinfect", "Clean dishwasher",
]

# Model display order and colours
MODEL_ORDER = ["TinyHAR", "AST", "TSMixer",
               "Late Fusion", "GMF", "MBT", "CMA", "LMF"]
MODEL_COLORS = {
    "TinyHAR":    "#2196F3",
    "AST":        "#FF5722",
    "TSMixer":    "#4CAF50",
    "Late Fusion":"#9C27B0",
    "GMF":        "#FF9800",
    "MBT":        "#E91E63",
    "CMA":        "#00BCD4",
    "LMF":        "#795548",
}

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


class ASTUnimodalModel(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.encoder    = ASTEncoder(embed_dim=embed_dim, freeze_backbone=True)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, audio_list):
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=audio_list[0].device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a
        return self.classifier(self.encoder(padded))


class FusionModel(nn.Module):
    def __init__(self, fusion_layer):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = fusion_layer

    def forward(self, imu, audio_list, humidity):
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        return self.fusion(self.tinyhar(imu), self.ast(padded), self.tsmixer(humidity))


def build_model(name):
    """Returns (model, modality_str, checkpoint_path)."""
    ckpt = lambda tag: os.path.join(MODELS_DIR, f"{tag}_fold{FOLD}.pt")

    if name == "TinyHAR":
        m = UnimodalModel(TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM),
                          EMBED_DIM, NUM_CLASSES)
        return m, "imu", ckpt("tinyhar")

    if name == "AST":
        return ASTUnimodalModel(EMBED_DIM, NUM_CLASSES), "audio", ckpt("ast")

    if name == "TSMixer":
        m = UnimodalModel(TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1,
                                         embed_dim=EMBED_DIM),
                          EMBED_DIM, NUM_CLASSES)
        return m, "humidity", ckpt("tsmixer")

    fusion_map = {
        "Late Fusion": (LateFusion(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, hidden_dim=256, dropout=0.3), "late_fusion"),
        "GMF": (GatedMultimodalFusion(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, hidden_dim=256, dropout=0.3), "gmf"),
        "MBT": (MultimodalBottleneckTransformer(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, model_dim=256, dropout=0.1), "mbt"),
        "CMA": (CrossModalAttention(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, hidden_dim=256, dropout=0.3), "cma"),
        "LMF": (LowRankMultimodalFusion(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, dropout=0.3), "lmf"),
    }
    fusion_layer, tag = fusion_map[name]
    return FusionModel(fusion_layer), "fusion", ckpt(tag)


# ── Inference ─────────────────────────────────────────────────────────────────

def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


@torch.no_grad()
def run_inference(model, loader, device, modality):
    model.eval()
    all_preds, all_labels = [], []
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        if modality == "imu":
            logits = model(imu)
        elif modality == "humidity":
            logits = model(humidity)
        elif modality == "audio":
            logits = model([a.to(device) for a in audio])
        elif modality == "fusion":
            logits = model(imu, audio, humidity)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_participant(pid, model, modality, device):
    ds = HARMESDataset([pid], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    preds, labels = run_inference(model, loader, device, modality)
    return {
        "n_windows":    len(ds),
        "macro_f1":     float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy":     float(accuracy_score(labels, preds)),
        "per_class_f1": f1_score(labels, preds, average=None,
                                 labels=list(range(NUM_CLASSES)), zero_division=0),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def _save(fig, filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def fig_lefthander_gap(all_results, models_run):
    """Bar chart: mean F1 of left-handers vs right-handers per model."""
    lh_means, rh_means, lh_stds, gaps = [], [], [], []
    labels = []
    for name in models_run:
        res = all_results[name]
        lh_f1 = [res[p]["macro_f1"] for p in LEFT_HANDERS  if p in res]
        rh_f1 = [res[p]["macro_f1"] for p in RIGHT_HANDERS if p in res]
        if not lh_f1 or not rh_f1: continue
        lh_means.append(np.mean(lh_f1))
        rh_means.append(np.mean(rh_f1))
        lh_stds.append(np.std(lh_f1))
        gaps.append(np.mean(rh_f1) - np.mean(lh_f1))
        labels.append(name)

    xs = np.arange(len(labels)); width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Left-hander Impact by Model — Fold 1 (P10, P14 vs P8,P9,P11,P12,P13)",
                 fontweight="bold")

    # Left: side-by-side F1 bars
    b1 = ax1.bar(xs - width/2, rh_means, width, label="Right-handed (P8,9,11,12,13)",
                 color="#4CAF50", alpha=0.85)
    b2 = ax1.bar(xs + width/2, lh_means, width, label="Left-handed (P10, P14)",
                 color="#F44336", alpha=0.85, yerr=lh_stds, capsize=4)
    for bar, v in list(zip(b1, rh_means)) + list(zip(b2, lh_means)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Macro F1"); ax1.set_ylim(0, 1)
    ax1.axhline(1/NUM_CLASSES, color="gray", linestyle="--", linewidth=1, label="Random")
    ax1.set_title("F1 by Handedness Group"); ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Right: gap bars (smaller = better at handling left-handers)
    colors = [MODEL_COLORS.get(l, "#607D8B") for l in labels]
    bars = ax2.bar(xs, gaps, color=colors, alpha=0.85)
    for bar, g in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{g:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("F1 Gap (right − left)"); ax2.set_ylim(0, max(gaps) * 1.35 + 0.02)
    ax2.set_title("Left-hander Penalty (lower = better)")
    ax2.grid(axis="y", alpha=0.3)
    # Annotate the best fusion
    fusion_gaps = [(g, l) for g, l in zip(gaps, labels) if l not in ("TinyHAR","AST","TSMixer")]
    if fusion_gaps:
        best_gap, best_name = min(fusion_gaps)
        idx = labels.index(best_name)
        ax2.annotate("Best fusion", xy=(idx, best_gap),
                     xytext=(idx + 0.5, best_gap + 0.03),
                     fontsize=8, color="green", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="green"))

    plt.tight_layout()
    _save(fig, "lefthander_gap.png")


def fig_participant_heatmap(all_results, models_run):
    """Heatmap: F1 per participant × model."""
    data = np.zeros((len(models_run), len(FOLD1_ALL)))
    for i, name in enumerate(models_run):
        for j, pid in enumerate(FOLD1_ALL):
            if pid in all_results[name]:
                data[i, j] = all_results[name][pid]["macro_f1"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = sns.heatmap(data, ax=ax,
                     xticklabels=[f"P{p}" for p in FOLD1_ALL],
                     yticklabels=models_run,
                     cmap="RdYlGn", vmin=0, vmax=1,
                     annot=True, fmt=".3f", annot_kws={"size": 9},
                     linewidths=0.5, linecolor="lightgray",
                     cbar_kws={"shrink": 0.8, "label": "Macro F1"})
    # Highlight left-hander columns
    for j, pid in enumerate(FOLD1_ALL):
        if pid in LEFT_HANDERS:
            ax.add_patch(plt.Rectangle((j, 0), 1, len(models_run),
                                       fill=False, edgecolor="red", lw=2.5))
    ax.set_title("Macro F1 per Participant × Model — Fold 1  (red = likely left-handed)",
                 fontweight="bold")
    ax.set_xlabel("Participant"); ax.set_ylabel("Model")
    plt.tight_layout()
    _save(fig, "per_participant_heatmap.png")


def fig_per_class_heatmap(all_results, models_run, pid, filename):
    """Heatmap: per-class F1 for one participant across all models."""
    data = np.zeros((len(models_run), NUM_CLASSES))
    for i, name in enumerate(models_run):
        if pid in all_results[name]:
            data[i] = all_results[name][pid]["per_class_f1"]

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(data, ax=ax,
                xticklabels=SHORT_NAMES, yticklabels=models_run,
                cmap="RdYlGn", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7.5},
                linewidths=0.4, linecolor="lightgray",
                cbar_kws={"shrink": 0.7, "label": "F1"})
    ax.set_title(f"Per-Class F1 — Participant P{pid} (likely left-handed) — All Models",
                 fontweight="bold")
    ax.set_xlabel("Activity"); ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    plt.tight_layout()
    _save(fig, filename)


# ── CSV writers ───────────────────────────────────────────────────────────────

def save_per_participant_csv(all_results, models_run):
    rows = []
    for name in models_run:
        for pid in FOLD1_ALL:
            if pid not in all_results[name]: continue
            r = all_results[name][pid]
            rows.append({
                "model":         name,
                "participant":   f"P{pid}",
                "handedness":    "left" if pid in LEFT_HANDERS else "right",
                "macro_f1":      f"{r['macro_f1']:.4f}",
                "accuracy":      f"{r['accuracy']:.4f}",
                "n_windows":     r["n_windows"],
            })
    path = os.path.join(OUT_DIR, "per_participant_all_models.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","participant","handedness",
                                          "macro_f1","accuracy","n_windows"])
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ {path}")


def save_gap_summary_csv(all_results, models_run):
    rows = []
    for name in models_run:
        res = all_results[name]
        lh_f1 = [res[p]["macro_f1"] for p in LEFT_HANDERS  if p in res]
        rh_f1 = [res[p]["macro_f1"] for p in RIGHT_HANDERS if p in res]
        if not lh_f1 or not rh_f1: continue
        rows.append({
            "model":            name,
            "lh_mean_f1":       f"{np.mean(lh_f1):.4f}",
            "rh_mean_f1":       f"{np.mean(rh_f1):.4f}",
            "gap_rh_minus_lh":  f"{np.mean(rh_f1) - np.mean(lh_f1):.4f}",
            "lh_p10_f1":        f"{res.get(10, {}).get('macro_f1', float('nan')):.4f}",
            "lh_p14_f1":        f"{res.get(14, {}).get('macro_f1', float('nan')):.4f}",
        })
    # Sort by gap ascending (best fusion = smallest gap)
    rows.sort(key=lambda r: float(r["gap_rh_minus_lh"]))
    path = os.path.join(OUT_DIR, "lefthander_gap_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","lh_mean_f1","rh_mean_f1",
                                          "gap_rh_minus_lh","lh_p10_f1","lh_p14_f1"])
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ {path}")


def save_per_class_csv(all_results, models_run, pid, filename):
    rows = []
    for name in models_run:
        if pid not in all_results[name]: continue
        for cls_id, (cls_name, f1) in enumerate(
                zip(SHORT_NAMES, all_results[name][pid]["per_class_f1"])):
            rows.append({"model": name, "class_id": cls_id,
                         "activity": cls_name, "f1": f"{f1:.4f}"})
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","class_id","activity","f1"])
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_MODEL_NAMES = ["TinyHAR", "AST", "TSMixer", "Late Fusion", "GMF", "MBT", "CMA", "LMF"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Which models to run. Options: all | unimodal | fusion | <name> ...")
    args = parser.parse_args()

    # Resolve model list
    if args.models == ["all"]:
        requested = ALL_MODEL_NAMES
    elif args.models == ["unimodal"]:
        requested = ["TinyHAR", "AST", "TSMixer"]
    elif args.models == ["fusion"]:
        requested = ["Late Fusion", "GMF", "MBT", "CMA", "LMF"]
    else:
        requested = args.models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {requested}")
    print(f"Fold {FOLD} participants: {FOLD1_ALL}")
    print(f"Likely left-handers: P10, P14\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    models_run  = []

    for name in requested:
        print(f"{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        try:
            model, modality, ckpt_path = build_model(name)
        except Exception as e:
            print(f"  ✗ Build failed: {e}"); continue

        if not os.path.exists(ckpt_path):
            print(f"  ✗ Checkpoint not found: {ckpt_path} — skipping"); continue

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)

        results = {}
        print(f"  {'P':>3}  {'F1':>7}  {'Acc':>7}  {'Windows':>8}  flag")
        print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*8}  ----")
        for pid in FOLD1_ALL:
            r = evaluate_participant(pid, model, modality, device)
            if r is None: continue
            results[pid] = r
            flag = "  ← LEFT" if pid in LEFT_HANDERS else ""
            print(f"  P{pid:>2}  {r['macro_f1']:>7.4f}  {r['accuracy']:>7.4f}  "
                  f"{r['n_windows']:>8,}{flag}")

        lh = [results[p]["macro_f1"] for p in LEFT_HANDERS  if p in results]
        rh = [results[p]["macro_f1"] for p in RIGHT_HANDERS if p in results]
        if lh and rh:
            print(f"\n  Right-handers mean: {np.mean(rh):.4f}  "
                  f"Left-handers mean: {np.mean(lh):.4f}  "
                  f"Gap: {np.mean(rh)-np.mean(lh):+.4f}")

        all_results[name] = results
        models_run.append(name)
        print()

    if not models_run:
        print("No models ran successfully — check checkpoints exist."); return

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving outputs...")
    print(f"{'='*60}")

    save_per_participant_csv(all_results, models_run)
    save_gap_summary_csv(all_results, models_run)
    for pid in LEFT_HANDERS:
        save_per_class_csv(all_results, models_run, pid, f"p{pid}_per_class.csv")

    fig_lefthander_gap(all_results, models_run)
    fig_participant_heatmap(all_results, models_run)
    for pid in LEFT_HANDERS:
        fig_per_class_heatmap(all_results, models_run, pid, f"p{pid}_per_class_heatmap.png")

    # ── Final ranking ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LEFT-HANDER GAP RANKING (smaller gap = handles left-handers better)")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'RH mean':>9} {'LH mean':>9} {'Gap':>8}  {'P10':>7}  {'P14':>7}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}")

    ranking = []
    for name in models_run:
        res = all_results[name]
        lh = [res[p]["macro_f1"] for p in LEFT_HANDERS  if p in res]
        rh = [res[p]["macro_f1"] for p in RIGHT_HANDERS if p in res]
        if lh and rh:
            ranking.append((np.mean(rh) - np.mean(lh), name,
                            np.mean(rh), np.mean(lh),
                            res.get(10, {}).get("macro_f1", float("nan")),
                            res.get(14, {}).get("macro_f1", float("nan"))))

    for gap, name, rh_m, lh_m, p10, p14 in sorted(ranking):
        print(f"  {name:<12}  {rh_m:>7.4f}  {lh_m:>7.4f}  {gap:>+7.4f}"
              f"  {p10:>7.4f}  {p14:>7.4f}")

    print(f"\nOutputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
