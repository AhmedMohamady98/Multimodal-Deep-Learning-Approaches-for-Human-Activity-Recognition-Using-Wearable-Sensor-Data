"""
Save raw GMF LOPO predictions to a master CSV.

Loads lopo_pXX_best.pt for each participant, runs inference on that
participant's held-out windows (10s, no overlap), and saves per-window:
  - participant      : int (1–20)
  - window_idx       : int (0-based within participant)
  - true_label       : int (0–15, includes Null at 15)
  - predicted_label  : int (0–15, includes Null at 15)
  - true_activity    : str (human-readable)
  - predicted_activity: str (human-readable)
  - correct          : bool
  - prob_class_00 … prob_class_15 : float (softmax probabilities, 16 classes)

Output:
  results/v3/10s/lopo/lopo_raw_predictions.csv   (~16k rows)

Usage:
    PYTHONPATH=. python jobs/save_lopo_predictions.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.gmf import GatedMultimodalFusion

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
CHECKPOINT_DIR = "results/v3/10s/lopo/models"
OUT_DIR        = "results/v3/10s/lopo"
OUT_CSV        = os.path.join(OUT_DIR, "lopo_raw_predictions.csv")

WINDOW_SIZE = 500   # 10s @ 50 Hz
STEP        = 500   # no overlap
EMBED_DIM   = 128
BATCH_SIZE  = 32
NUM_WORKERS = 4

ALL_PARTICIPANTS = list(range(1, 21))


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
def run_inference_with_probs(model, loader, device):
    """Returns (preds, labels, probs) — probs shape (N, NUM_CLASSES)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        logits   = model(imu, audio, humidity)
        probs    = F.softmax(logits, dim=-1)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs, axis=0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []

    for p in ALL_PARTICIPANTS:
        ckpt = os.path.join(CHECKPOINT_DIR, f"lopo_p{p:02d}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  SKIP P{p:02d} — checkpoint missing: {ckpt}")
            continue

        print(f"Saving predictions for P{p:02d} ...")
        test_ds = HARMESDataset([p], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
        loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

        model = GMFModel().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        preds, labels, probs = run_inference_with_probs(model, loader, device)

        for i, (pred, label, prob) in enumerate(zip(preds, labels, probs)):
            row = {
                "participant":          p,
                "window_idx":           i,
                "true_label":           int(label),
                "predicted_label":      int(pred),
                "true_activity":        ID_TO_ACTIVITY[int(label)],
                "predicted_activity":   ID_TO_ACTIVITY[int(pred)],
                "correct":              bool(pred == label),
            }
            for c in range(NUM_CLASSES):
                row[f"prob_class_{c:02d}"] = float(prob[c])
            all_rows.append(row)

        n = len(labels)
        acc = float((preds == labels).mean())
        print(f"  P{p:02d}: {n} windows, acc={acc:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    total = len(df)
    correct = df["correct"].sum()
    print(f"\nSaved {OUT_CSV}")
    print(f"  Total rows : {total:,}")
    print(f"  Participants: {df['participant'].nunique()}/20")
    print(f"  Overall acc : {correct/total:.4f}")
    print(f"  Columns     : {list(df.columns)}")


if __name__ == "__main__":
    main()
