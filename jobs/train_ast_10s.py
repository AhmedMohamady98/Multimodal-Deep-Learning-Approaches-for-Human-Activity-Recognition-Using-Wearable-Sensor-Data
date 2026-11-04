"""
Train AST (Audio) at 10-second windows — single fold, pass --fold 0/1/2.

10s windows perfectly match AudioSet pretraining clip length (10s).
ASTFeatureExtractor pads to 1024 mel-frames ≈ 10s → 100% spectrogram fill.
This is the optimal configuration for AST.

Run via separate SLURM jobs per fold:
  sbatch jobs/train_ast_10s_fold0.sh
  sbatch jobs/train_ast_10s_fold1.sh
  sbatch jobs/train_ast_10s_fold2.sh

Results saved to: results/unimodal_models/
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import HARMESDataset, NUM_CLASSES
from src.data.splits import get_3fold_splits_with_val
from src.encoders.ast.model import ASTEncoder
from src.training.trainer import train_model

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
RESULTS_DIR = "results/unimodal_models"
BATCH_SIZE  = 32   # same as 2s/8s — AST spectrogram shape is always 1024 frames
EMBED_DIM   = 128
NUM_WORKERS = 4

# ── 10-second window parameters ─────────────────────────────────────────────
WINDOW_SIZE = 500   # 10s × 50 Hz  (audio: ~10s × 44100 Hz ≈ 441,000 samples)
STEP        = 250   # 50% overlap
# Note: ASTFeatureExtractor pads all audio to 1024 mel-frames (≈10s).
# 10s audio fills 1024/1024 frames = 100% real content — perfectly matches
# AudioSet pretraining distribution. This is the ideal window size for AST.


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


def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2],
                        help="Fold index to train (0, 1, or 2)")
    args = parser.parse_args()
    fold_idx = args.fold

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE / 50:.0f}s  |  "
          f"Step: {STEP} samples = {STEP / 50:.0f}s (50% overlap)")
    print(f"AST spectrogram fill: ~100% (10s matches AudioSet pretraining)")

    splits = get_3fold_splits_with_val(list(range(1, 21)))
    fold   = splits[fold_idx]

    print(f"\n{'='*60}")
    print(f"AST (10s) — Fold {fold_idx}")
    print(f"  train={fold['train']}")
    print(f"  val  ={fold['val']}")
    print(f"  test ={fold['test']}")
    print(f"{'='*60}")

    train_ds = HARMESDataset(fold["train"], DATA_DIR,
                             window_size=WINDOW_SIZE, step=STEP)
    val_ds   = HARMESDataset(fold["val"],   DATA_DIR,
                             window_size=WINDOW_SIZE, step=STEP)
    test_ds  = HARMESDataset(fold["test"],  DATA_DIR,
                             window_size=WINDOW_SIZE, step=STEP)
    print(f"Train windows: {len(train_ds):,}  |  Val windows: {len(val_ds):,}  |  Test windows: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True)

    model = ASTModel(EMBED_DIM, NUM_CLASSES).to(device)

    ckpt = os.path.join(RESULTS_DIR, f"ast_fold{fold_idx}.pt")
    test_f1, test_acc, best_val_f1 = train_model(
        model, train_loader, val_loader, test_loader,
        modality="audio", device=device,
        epochs=15, lr=1e-3, patience=4,
        checkpoint_path=ckpt,
    )

    print(f"\n{'='*60}")
    print(f"AST 10s Fold {fold_idx} FINAL TEST: F1={test_f1:.4f}  acc={test_acc:.4f}  "
          f"(best val_F1={best_val_f1:.4f})")
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
