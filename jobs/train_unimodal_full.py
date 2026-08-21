"""
Train all 3 unimodal baselines on the full 20-participant dataset.
3-fold group cross-validation (~6-7 participants per fold).
Best model weights saved to results/v3/10s/unimodal_models/.

Run via: sbatch jobs/train_unimodal_full.sh
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
from src.encoders.ast.model import ASTEncoder
from src.training.trainer import train_model

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
RESULTS_DIR     = "results/v3/10s/unimodal_models"
PARTICIPANTS    = list(range(1, 21))   # all 20
BATCH_SIZE      = 64
EPOCHS_DEFAULT  = 50   # TinyHAR, TSMixer
EPOCHS_AST      = 15   # AST frozen backbone — converges fast
EMBED_DIM       = 128
NUM_WORKERS     = 4
WINDOW_SIZE     = 500   # 10s × 50 Hz — matches the 10s training campaign
STEP            = 250   # 50% overlap (matches 3-fold CV training)


# ── Unimodal model wrapper ────────────────────────────────────────────────────

class UnimodalModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return (
        torch.stack(imus),
        list(audios),          # variable length — keep as list
        torch.stack(hums),
        torch.stack(labels),
    )


# ── Model definitions ─────────────────────────────────────────────────────────

def make_tinyhar():
    encoder = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
    return UnimodalModel(encoder, EMBED_DIM, NUM_CLASSES), "imu"


def make_tsmixer():
    encoder = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
    return UnimodalModel(encoder, EMBED_DIM, NUM_CLASSES), "humidity"


def make_ast():
    encoder = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)

    class ASTModel(nn.Module):
        def __init__(self, encoder, num_classes):
            super().__init__()
            self.encoder    = encoder
            self.classifier = nn.Linear(EMBED_DIM, num_classes)

        def forward(self, audio_list):
            max_len = max(a.shape[0] for a in audio_list)
            padded  = torch.zeros(len(audio_list), max_len,
                                  device=audio_list[0].device)
            for i, a in enumerate(audio_list):
                padded[i, :a.shape[0]] = a
            return self.classifier(self.encoder(padded))

    return ASTModel(encoder, NUM_CLASSES), "audio"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(name, make_fn, splits, device, epochs=50, patience=10):
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")

    fold_f1s, fold_accs = [], []
    model_tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    for fold_idx, fold in enumerate(splits):
        print(f"\n--- Fold {fold_idx} ---")
        print(f"  train={fold['train']}")
        print(f"  val  ={fold['val']}")
        print(f"  test ={fold['test']}")

        train_ds = HARMESDataset(fold["train"], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
        val_ds   = HARMESDataset(fold["val"],   DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
        test_ds  = HARMESDataset(fold["test"],  DATA_DIR, window_size=WINDOW_SIZE, step=STEP)

        print(f"  Train windows: {len(train_ds):,}  |  Val windows: {len(val_ds):,}  |  Test windows: {len(test_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  collate_fn=collate_fn,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=NUM_WORKERS, pin_memory=True)

        model, modality = make_fn()
        model = model.to(device)

        ckpt_path = os.path.join(RESULTS_DIR, f"{model_tag}_fold{fold_idx}.pt")

        test_f1, test_acc, best_val_f1 = train_model(
            model, train_loader, val_loader, test_loader,
            modality=modality, device=device,
            epochs=epochs, lr=1e-3, patience=patience,
            checkpoint_path=ckpt_path,
        )

        print(f"  Fold {fold_idx} FINAL TEST: F1={test_f1:.4f}  acc={test_acc:.4f}  "
              f"(best val_F1={best_val_f1:.4f})")
        fold_f1s.append(test_f1)
        fold_accs.append(test_acc)

    mean_f1  = np.mean(fold_f1s)
    std_f1   = np.std(fold_f1s)
    mean_acc = np.mean(fold_accs)
    print(f"\n{name} FINAL: F1={mean_f1:.4f} ± {std_f1:.4f}  acc={mean_acc:.4f}")
    return mean_f1, std_f1, mean_acc, fold_f1s, fold_accs


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE/50:.0f}s  |  Step: {STEP} (50% overlap)")

    splits = get_3fold_splits_with_val(PARTICIPANTS)
    print(f"\n3-fold split:")
    for i, fold in enumerate(splits):
        print(f"  Fold {i}: train={fold['train']}  val={fold['val']}  test={fold['test']}")

    results = {}
    results["TinyHAR (IMU)"]      = run_model("TinyHAR (IMU)",      make_tinyhar, splits, device, epochs=EPOCHS_DEFAULT, patience=10)
    results["TSMixer (Humidity)"] = run_model("TSMixer (Humidity)", make_tsmixer, splits, device, epochs=EPOCHS_DEFAULT, patience=10)
    results["AST (Audio)"]        = run_model("AST (Audio)",        make_ast,     splits, device, epochs=EPOCHS_AST,     patience=4)

    print(f"\n{'='*60}")
    print("UNIMODAL BASELINE SUMMARY (20 participants, 3-fold CV)")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Macro F1':>10} {'± std':>8} {'Accuracy':>10}")
    print("-" * 55)
    for name, (f1, std, acc, folds_f1, folds_acc) in results.items():
        print(f"{name:<25} {f1:>10.4f} {std:>8.4f} {acc:>10.4f}")
        for i, (ff1, facc) in enumerate(zip(folds_f1, folds_acc)):
            print(f"  {'Fold '+str(i):<23} {ff1:>10.4f} {'':>8} {facc:>10.4f}")

    print(f"\nCheckpoints saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
