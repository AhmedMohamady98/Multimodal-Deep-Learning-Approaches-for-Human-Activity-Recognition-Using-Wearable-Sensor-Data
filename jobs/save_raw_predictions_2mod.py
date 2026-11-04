"""
Dump per-window raw softmax predictions for the 2-modality (IMU + Audio)
ablation models — Late Fusion / GMF / CMA, all 3 folds.

Mirror of jobs/save_raw_predictions.py but for the 2-mod ablation
variants. Output goes to:
  results/predictions/2mod_all_models_raw_predictions.csv.gz
  results/predictions/{method}_2mod_fold{i}_raw_preds.csv

These CSVs are consumed by the ablation analysis notebooks (humidity
contribution, 3-mod vs 2-mod comparisons).

Schema matches save_raw_predictions.py exactly so notebooks can read
both 3-mod and 2-mod predictions through one parser.

Usage:
  PYTHONPATH=. python jobs/save_raw_predictions_2mod.py

Submit via SLURM:
  sbatch jobs/save_raw_predictions_2mod.sh
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.data.splits import get_3fold_splits

# Reuse model classes + collate from the training script
from jobs.train_ablation_2mod_10s import (
    FusionModel2Mod,
    FUSION_CLASSES_2MOD,
    collate_fn,
    WINDOW_SIZE,
    STEP,
)

# ── Config ────────────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR = os.path.join(ROOT, "results", "models")
OUT_DIR    = os.path.join(ROOT, "results", "predictions")

BATCH_SIZE  = 64
NUM_WORKERS = 2

CLASS_NAMES = [ID_TO_ACTIVITY[i] for i in range(NUM_CLASSES)]


def ckpt_path(method: str, fold: int) -> str:
    return os.path.join(MODELS_DIR, f"{method}_2mod_fold{fold}_best.pt")


def build_model(method: str) -> torch.nn.Module:
    fusion_cls = FUSION_CLASSES_2MOD[method]
    return FusionModel2Mod(fusion_cls())


# ── Inference per participant ─────────────────────────────────────────────────

@torch.no_grad()
def run_participant(model, participant_id, device):
    ds = HARMESDataset([participant_id], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    if len(ds) == 0:
        return np.array([]), np.array([]), np.zeros((0, NUM_CLASSES), dtype=np.float32)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model.eval()
    all_true, all_pred, all_probs = [], [], []
    for imu, audio_list, _humidity, labels in loader:
        imu   = imu.to(device)
        logits = model(imu, audio_list)         # 2-mod model ignores humidity
        probs  = F.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_probs, axis=0))


# ── CSV plumbing (schema mirrors save_raw_predictions.py) ─────────────────────

PROB_COLS = [f"prob_{i}" for i in range(NUM_CLASSES)]
HEADER    = (["participant_id", "window_idx", "fold", "model",
              "true_label", "true_label_name",
              "pred_label", "pred_label_name", "correct"]
             + PROB_COLS)


def rows_for_participant(participant_id, fold, model_key, true_labels, pred_labels, probs):
    rows = []
    for w_idx, (tl, pl, pr) in enumerate(zip(true_labels, pred_labels, probs)):
        row = [
            participant_id, w_idx, fold, model_key,
            int(tl), CLASS_NAMES[int(tl)],
            int(pl), CLASS_NAMES[int(pl)],
            int(tl == pl),
        ] + [f"{p:.6f}" for p in pr]
        rows.append(row)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits  = get_3fold_splits(list(range(1, 21)))

    METHODS = sorted(FUSION_CLASSES_2MOD)   # ["cma", "gmf", "late_fusion"]

    # Master CSV — every prediction from every 2-mod model across all folds
    master_path = os.path.join(OUT_DIR, "2mod_all_models_raw_predictions.csv.gz")
    master_fh   = open(master_path, "w", newline="")
    master_w    = csv.writer(master_fh)
    master_w.writerow(HEADER)

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Model: {method} (2-mod, IMU+Audio)")

        for fold_idx, split in enumerate(splits):
            ckpt = ckpt_path(method, fold_idx)
            if not os.path.exists(ckpt):
                print(f"  Fold {fold_idx}: checkpoint not found at {ckpt} — skipping")
                continue

            print(f"  Fold {fold_idx}: loading {ckpt}")
            model = build_model(method).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            if isinstance(state, dict):
                for k in ("model_state_dict", "state_dict", "model_state"):
                    if k in state:
                        state = state[k]
                        break
            model.load_state_dict(state)

            # Per-method-per-fold CSV alongside the master file
            per_path = os.path.join(OUT_DIR, f"{method}_2mod_fold{fold_idx}_raw_preds.csv")
            with open(per_path, "w", newline="") as per_fh:
                per_w = csv.writer(per_fh)
                per_w.writerow(HEADER)

                for participant_id in sorted(split["test"]):
                    print(f"    Participant {participant_id:02d} …", end="", flush=True)
                    true_l, pred_l, probs = run_participant(model, participant_id, device)
                    n = len(true_l)
                    if n == 0:
                        print(" (no windows)")
                        continue

                    rows = rows_for_participant(participant_id, fold_idx, f"{method}_2mod",
                                                true_l, pred_l, probs)
                    per_w.writerows(rows)
                    master_w.writerow([])   # safe no-op separator; csv ignores empties
                    master_w.writerows(rows)
                    correct = sum(int(tl == pl) for tl, pl in zip(true_l, pred_l))
                    print(f"  {n:>4} windows, {correct/n:.1%} correct")

            print(f"  → {per_path}")

    master_fh.close()
    print(f"\nMaster CSV → {master_path}")


if __name__ == "__main__":
    main()
