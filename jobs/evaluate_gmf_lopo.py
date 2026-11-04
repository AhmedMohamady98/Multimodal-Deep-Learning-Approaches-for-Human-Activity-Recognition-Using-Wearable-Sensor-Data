"""
Evaluate all 20 LOPO checkpoints and aggregate results.

Loads lopo_pXX_best.pt for each participant, runs inference on that
participant's windows (10s, no overlap), and saves:
  - results/lopo/lopo_results.json   — per-participant metrics
  - results/lopo/lopo_summary.csv    — summary table for thesis

Usage:
    PYTHONPATH=. python jobs/evaluate_gmf_lopo.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.data.dataset import HARMESDataset, NUM_CLASSES
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.gmf import GatedMultimodalFusion

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
CHECKPOINT_DIR = "results/lopo/models"
OUT_DIR        = "results/lopo"

WINDOW_SIZE = 500   # 10s @ 50 Hz
STEP        = 500   # no overlap
EMBED_DIM   = 128
BATCH_SIZE  = 32
NUM_WORKERS = 4

ALL_PARTICIPANTS = list(range(1, 21))

ACTIVITY_NAMES = [
    "Washing hands", "Floor cleaning", "Window cleaning", "Vacuum cleaning",
    "Brushing teeth", "Washing dishes", "Putting away dishes", "Making tea",
    "Cutting vegetables", "Creaming hands", "Drinking", "Watering plants",
    "Cleaning table", "Disinfecting hands", "Cleaning dishwasher",
]


# ── Model ─────────────────────────────────────────────────────────────────────

class GMFModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = GatedMultimodalFusion(
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
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        logits   = model(imu, audio, humidity)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    all_preds_global, all_labels_global = [], []

    for p in ALL_PARTICIPANTS:
        ckpt = os.path.join(CHECKPOINT_DIR, f"lopo_p{p:02d}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  SKIP P{p:02d} — checkpoint missing: {ckpt}")
            continue

        print(f"\nEvaluating P{p:02d} ...")
        test_ds = HARMESDataset([p], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
        loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

        model = GMFModel().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        preds, labels = run_inference(model, loader, device)
        all_preds_global.append(preds)
        all_labels_global.append(labels)

        f1  = f1_score(labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(labels, preds)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
        cm  = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

        print(f"  P{p:02d}: F1={f1:.4f}  acc={acc:.4f}  n_windows={len(labels)}")
        results.append({
            "participant": p,
            "n_windows":   int(len(labels)),
            "macro_f1":    float(f1),
            "accuracy":    float(acc),
            "per_class_f1": [float(x) for x in per_class_f1],
            "confusion_matrix": cm.tolist(),
        })

    # Aggregate across all participants
    all_preds  = np.concatenate(all_preds_global)
    all_labels = np.concatenate(all_labels_global)
    mean_f1    = float(np.mean([r["macro_f1"] for r in results]))
    std_f1     = float(np.std([r["macro_f1"] for r in results], ddof=1))
    mean_acc   = float(np.mean([r["accuracy"] for r in results]))
    global_f1  = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    global_acc = float(accuracy_score(all_labels, all_preds))

    summary = {
        "n_participants": len(results),
        "mean_macro_f1":  mean_f1,
        "std_macro_f1":   std_f1,
        "mean_accuracy":  mean_acc,
        "global_macro_f1": global_f1,
        "global_accuracy": global_acc,
        "window_size_s":   WINDOW_SIZE / 50,
        "step_overlap":    "none",
        "protocol":        "LOPO",
        "per_participant": results,
    }

    out_json = os.path.join(OUT_DIR, "lopo_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    # Summary CSV
    rows = [{"participant": r["participant"],
             "n_windows":   r["n_windows"],
             "macro_f1":    round(r["macro_f1"], 4),
             "accuracy":    round(r["accuracy"], 4)} for r in results]
    rows.append({"participant": "MEAN ± std",
                 "n_windows": sum(r["n_windows"] for r in results),
                 "macro_f1":  f"{mean_f1:.4f} ± {std_f1:.4f}",
                 "accuracy":  f"{mean_acc:.4f}"})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "lopo_summary.csv"), index=False)

    print(f"\n{'='*50}")
    print(f"LOPO GMF — FINAL RESULTS")
    print(f"  Participants evaluated: {len(results)}/20")
    print(f"  Mean macro F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Global macro F1 (pooled): {global_f1:.4f}")
    print(f"  Mean accuracy:  {mean_acc:.4f}")
    print(f"  HARMES paper baseline: F1 = 0.760")
    print(f"  Delta vs paper: {mean_f1 - 0.760:+.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
