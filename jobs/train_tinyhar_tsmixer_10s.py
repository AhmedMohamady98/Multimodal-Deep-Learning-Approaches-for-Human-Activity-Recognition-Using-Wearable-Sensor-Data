"""
Train TinyHAR (IMU) + TSMixer (Humidity) at 10-second windows.

10s = 500 samples at 50 Hz, 50% overlap (step=250).
AudioSet clips are 10s → AST spectrogram fills 100% at this window size.

Results saved to: results/unimodal_models/
Compare with:     results/8s/models/  and  results/models/
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import HARMESDataset, NUM_CLASSES
from src.data.splits import get_3fold_splits_with_val
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.training.trainer import train_model

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
RESULTS_DIR = "results/unimodal_models"
BATCH_SIZE  = 64
EMBED_DIM   = 128
NUM_WORKERS = 4

# ── 10-second window parameters ─────────────────────────────────────────────
WINDOW_SIZE = 500   # 10s × 50 Hz
STEP        = 250   # 50% overlap


class UnimodalModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


# ════════════════════════════════════════════════════════════════════════════
#  TinyHAR — IMU (12 channels, both wrists)
# ════════════════════════════════════════════════════════════════════════════

def train_tinyhar(splits, device):
    print("\n" + "═" * 70)
    print("  TinyHAR (IMU) — 10s windows")
    print("═" * 70)

    fold_f1s, fold_accs = [], []

    for fold_idx, fold in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"TinyHAR — Fold {fold_idx}")
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

        # TinyHAR is fully length-agnostic — no model changes needed
        model = UnimodalModel(
            TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM),
            EMBED_DIM, NUM_CLASSES
        ).to(device)

        ckpt = os.path.join(RESULTS_DIR, f"tinyhar_fold{fold_idx}.pt")
        test_f1, test_acc, best_val_f1 = train_model(
            model, train_loader, val_loader, test_loader,
            modality="imu", device=device,
            epochs=50, lr=1e-3, patience=10,
            checkpoint_path=ckpt,
        )
        print(f"Fold {fold_idx} FINAL TEST: F1={test_f1:.4f}  acc={test_acc:.4f}  "
              f"(best val_F1={best_val_f1:.4f})")
        fold_f1s.append(test_f1)
        fold_accs.append(test_acc)

    print(f"\n{'='*60}")
    print(f"TinyHAR 10s FINAL: F1={np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}"
          f"  acc={np.mean(fold_accs):.4f}")
    for i, (f1, acc) in enumerate(zip(fold_f1s, fold_accs)):
        print(f"  Fold {i}: F1={f1:.4f}  acc={acc:.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  TSMixer — Humidity (1 channel, BME280)
# ════════════════════════════════════════════════════════════════════════════

def train_tsmixer(splits, device):
    print("\n" + "═" * 70)
    print("  TSMixer (Humidity) — 10s windows")
    print("═" * 70)

    fold_f1s, fold_accs = [], []

    for fold_idx, fold in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"TSMixer — Fold {fold_idx}")
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

        # TSMixer: seq_len must match window size at construction time
        model = UnimodalModel(
            TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM),
            EMBED_DIM, NUM_CLASSES
        ).to(device)

        ckpt = os.path.join(RESULTS_DIR, f"tsmixer_fold{fold_idx}.pt")
        test_f1, test_acc, best_val_f1 = train_model(
            model, train_loader, val_loader, test_loader,
            modality="humidity", device=device,
            epochs=50, lr=1e-3, patience=10,
            checkpoint_path=ckpt,
        )
        print(f"Fold {fold_idx} FINAL TEST: F1={test_f1:.4f}  acc={test_acc:.4f}  "
              f"(best val_F1={best_val_f1:.4f})")
        fold_f1s.append(test_f1)
        fold_accs.append(test_acc)

    print(f"\n{'='*60}")
    print(f"TSMixer 10s FINAL: F1={np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}"
          f"  acc={np.mean(fold_accs):.4f}")
    for i, (f1, acc) in enumerate(zip(fold_f1s, fold_accs)):
        print(f"  Fold {i}: F1={f1:.4f}  acc={acc:.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE / 50:.0f}s  |  "
          f"Step: {STEP} samples = {STEP / 50:.0f}s (50% overlap)")

    splits = get_3fold_splits_with_val(list(range(1, 21)))

    train_tinyhar(splits, device)
    train_tsmixer(splits, device)

    print("\n" + "═" * 70)
    print("  All done — checkpoints in results/unimodal_models/")
    print("═" * 70)


if __name__ == "__main__":
    main()
