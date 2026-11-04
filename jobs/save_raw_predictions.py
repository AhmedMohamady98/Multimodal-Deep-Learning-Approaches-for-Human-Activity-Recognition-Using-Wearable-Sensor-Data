"""
Save raw per-window predictions for all models (unimodal + fusion).

For each model × fold × participant, saves:
  - participant_id     : int
  - window_idx         : int (0-based within this participant+fold)
  - fold               : int (0, 1, 2)
  - model              : str (e.g. "gmf", "tinyhar")
  - true_label         : int (0–15, includes Null at 15)
  - true_label_name    : str (e.g. "Washing hands")
  - pred_label         : int (0–15, includes Null at 15)
  - pred_label_name    : str
  - correct            : bool
  - prob_0 … prob_15   : float32 softmax probabilities (16 classes)

Outputs:
  results/predictions/
  ├── {model}_fold{i}_raw_preds.csv   ← one file per model × fold
  └── all_models_raw_predictions.csv.gz  ← master combined file

Usage:
  PYTHONPATH=. python jobs/save_raw_predictions.py
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR          = os.path.join(ROOT, "results", "models")
UNIMODAL_MODELS_DIR = os.path.join(ROOT, "results", "unimodal_models")
OUT_DIR    = os.path.join(ROOT, "results", "predictions")

WINDOW_SIZE = 500
STEP        = 250
EMBED_DIM   = 128
BATCH_SIZE  = 64
NUM_WORKERS = 2

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
from src.fusion.lmf import LowRankMultimodalFusion
from src.fusion.decision_fusion import DecisionFusion
from src.fusion.cls_transformer import CLSTransformerFusion

CLASS_NAMES = [ID_TO_ACTIVITY[i] for i in range(NUM_CLASSES)]

# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)

# ---------------------------------------------------------------------------
# Unimodal model wrappers
# ---------------------------------------------------------------------------
class TinyHARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.classifier  = nn.Linear(EMBED_DIM, NUM_CLASSES)
    def forward(self, imu, audio_list, humidity):
        return self.classifier(self.encoder(imu))

class TSMixerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.classifier  = nn.Linear(EMBED_DIM, NUM_CLASSES)
    def forward(self, imu, audio_list, humidity):
        return self.classifier(self.encoder(humidity))

class ASTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.classifier  = nn.Linear(EMBED_DIM, NUM_CLASSES)
    def forward(self, imu, audio_list, humidity):
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        return self.classifier(self.encoder(padded))

# ---------------------------------------------------------------------------
# Fusion model wrapper
# ---------------------------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, fusion_module):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = fusion_module

    def forward(self, imu, audio_list, humidity):
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        return self.fusion(self.tinyhar(imu), self.ast(padded), self.tsmixer(humidity))

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def build_model(key: str) -> nn.Module:
    """Build an uninitialised model for the given key."""
    if key == "tinyhar":
        return TinyHARModel()
    if key == "tsmixer":
        return TSMixerModel()
    if key == "ast":
        return ASTModel()
    if key == "late_fusion":
        return FusionModel(LateFusion(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES))
    if key == "gmf":
        return FusionModel(GatedMultimodalFusion(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES))
    if key == "mbt":
        return FusionModel(MultimodalBottleneckTransformer(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES))
    if key == "cma":
        return FusionModel(CrossModalAttention(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES))
    if key == "lmf":
        return FusionModel(LowRankMultimodalFusion(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES))
    if key == "decision_fusion":
        return FusionModel(DecisionFusion(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES))
    if key == "cls_transformer":
        return FusionModel(CLSTransformerFusion(EMBED_DIM, EMBED_DIM, EMBED_DIM, NUM_CLASSES))
    raise ValueError(f"Unknown model key: {key}")

def ckpt_path(key: str, fold: int) -> str:
    """Return path to best checkpoint.
    Unimodal uses unimodal_models/ with modality suffix (e.g. tinyhar_imu_fold0.pt).
    Fusion uses models/ with _best suffix (e.g. gmf_fold0_best.pt).
    """
    _UNIMODAL_SUFFIX = {"tinyhar": "imu", "ast": "audio", "tsmixer": "humidity"}
    if key in _UNIMODAL_SUFFIX:
        suffix = _UNIMODAL_SUFFIX[key]
        p = os.path.join(UNIMODAL_MODELS_DIR, f"{key}_{suffix}_fold{fold}.pt")
    else:
        p = os.path.join(MODELS_DIR, f"{key}_fold{fold}_best.pt")
    return p

# Methods that return log-probabilities instead of raw logits
LOG_PROB_MODELS = {"decision_fusion"}

# ---------------------------------------------------------------------------
# Inference — per participant
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_participant(model, participant_id, device):
    """
    Run inference on a single participant. Returns:
      true_labels : (N,) int
      pred_labels : (N,) int
      probs       : (N, 15) float32
    """
    ds = HARMESDataset([participant_id], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    if len(ds) == 0:
        return np.array([]), np.array([]), np.zeros((0, NUM_CLASSES), dtype=np.float32)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model.eval()
    all_true, all_pred, all_probs = [], [], []

    for imu, audio_list, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)

        out = model(imu, audio_list, humidity)   # (B, C) logits or log-probs

        # Convert to probabilities
        if isinstance(model, FusionModel) and isinstance(model.fusion, DecisionFusion):
            probs = torch.exp(out)               # log-probs → probs
        else:
            probs = F.softmax(out, dim=-1)       # logits → probs

        preds = probs.argmax(dim=-1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_probs, axis=0))

# ---------------------------------------------------------------------------
# CSV writer helpers
# ---------------------------------------------------------------------------
PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]
HEADER    = (["participant_id", "window_idx", "fold", "model",
              "true_label", "true_label_name",
              "pred_label", "pred_label_name", "correct"]
             + PROB_COLS)

def rows_for_participant(participant_id, fold, model_key, true_labels, pred_labels, probs):
    rows = []
    for w_idx, (tl, pl, pr) in enumerate(zip(true_labels, pred_labels, probs)):
        row = [
            participant_id,
            w_idx,
            fold,
            model_key,
            int(tl),
            CLASS_NAMES[int(tl)],
            int(pl),
            CLASS_NAMES[int(pl)],
            int(tl == pl),
        ] + [f"{p:.6f}" for p in pr]
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits  = get_3fold_splits(list(range(1, 21)))

    ALL_MODELS = [
        "tinyhar", "ast", "tsmixer",
        "late_fusion", "gmf", "mbt", "cma", "lmf", "decision_fusion", "cls_transformer",
    ]

    master_path = os.path.join(OUT_DIR, "all_models_raw_predictions.csv.gz")
    master_fh   = open(master_path, "w", newline="")
    master_w    = csv.writer(master_fh)
    master_w.writerow(HEADER)

    for model_key in ALL_MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")

        for fold_idx, split in enumerate(splits):
            ckpt = ckpt_path(model_key, fold_idx)
            if not os.path.exists(ckpt):
                print(f"  Fold {fold_idx}: checkpoint not found at {ckpt} — skipping")
                continue

            print(f"  Fold {fold_idx}: loading {ckpt}")
            model = build_model(model_key).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            # Handle nested state dicts
            if isinstance(state, dict):
                for key in ("model_state_dict", "state_dict", "model_state"):
                    if key in state:
                        state = state[key]
                        break
            model.load_state_dict(state)

            # Per-model-fold CSV
            fold_csv_path = os.path.join(OUT_DIR, f"{model_key}_fold{fold_idx}_raw_preds.csv")
            fold_fh = open(fold_csv_path, "w", newline="")
            fold_w  = csv.writer(fold_fh)
            fold_w.writerow(HEADER)

            test_participants = split["test"]
            total_windows = 0

            for pid in test_participants:
                true_labels, pred_labels, probs = run_participant(model, pid, device)
                if len(true_labels) == 0:
                    continue

                rows = rows_for_participant(pid, fold_idx, model_key,
                                            true_labels, pred_labels, probs)
                fold_w.writerows(rows)
                master_w.writerows(rows)
                total_windows += len(rows)
                print(f"    P{pid:02d}: {len(rows)} windows")

            fold_fh.close()
            print(f"  Fold {fold_idx}: {total_windows} windows → {fold_csv_path}")

            # Free GPU memory between folds
            del model
            torch.cuda.empty_cache()

    master_fh.close()
    print(f"\n{'='*60}")
    print(f"Master file: {master_path}")
    print("Done.")


if __name__ == "__main__":
    main()
