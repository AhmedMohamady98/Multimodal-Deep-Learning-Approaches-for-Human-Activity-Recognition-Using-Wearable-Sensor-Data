"""
MBT (Multimodal Bottleneck Transformer) training — 10s windows.

Architecture:
    TinyHAR (IMU)      → 128-dim → Linear(128→256) → h_imu   (B,1,256) ─┐
    AST (Audio)        → 128-dim → Linear(128→256) → h_audio (B,1,256) ─┤ + 4 shared
    TSMixer (Humidity) → 128-dim → Linear(128→256) → h_hum   (B,1,256) ─┘   bottleneck tokens

    Repeated × 4 layers:
      imu_stream   = [h_imu,   bn] → TransformerLayer(8 heads) → [h_imu',   bn_imu]
      audio_stream = [h_audio, bn] → TransformerLayer(8 heads) → [h_audio', bn_audio]
      hum_stream   = [h_hum,   bn] → TransformerLayer(8 heads) → [h_hum',   bn_hum]
      bn = mean(bn_imu, bn_audio, bn_hum)   ← shared state updated each layer

    mean-pool(bn) → LayerNorm → Linear(256→16) → 16 classes

Initialization:
    TinyHAR, TSMixer, and the AST projection head are randomly initialised.
    The AST ViT backbone stays frozen on its AudioSet weights.
    All modalities are trained jointly from epoch 1.

Checkpointing:
    Two files are written per fold:
      mbt_fold{i}_best.pt   — best model weights (for evaluation)
      mbt_fold{i}_resume.pt — full training state (optimizer, scheduler,
                                      epoch, counters) so a timed-out job can
                                      resume exactly where it stopped.

Usage:
    PYTHONPATH=. python jobs/train_mbt_10s.py --fold 0
    PYTHONPATH=. python jobs/train_mbt_10s.py --fold 1
    PYTHONPATH=. python jobs/train_mbt_10s.py --fold 2
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.data.dataset import HARMESDataset, NUM_CLASSES
from src.data.splits import get_3fold_splits_with_val
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.mbt import MultimodalBottleneckTransformer

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
CHECKPOINT_DIR = "results/v3/10s/models"

WINDOW_SIZE = 500   # 10s × 50 Hz
STEP        = 250   # 50% overlap
EMBED_DIM   = 128
BATCH_SIZE  = 32
NUM_WORKERS = 4
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10


# ── Full fusion model ─────────────────────────────────────────────────────────

class MBTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = MultimodalBottleneckTransformer(
            imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, humidity_dim=EMBED_DIM,
            num_classes=NUM_CLASSES, model_dim=256, dropout=0.1,
        )

    def forward(self, imu, audio_list, humidity):
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        imu_emb   = self.tinyhar(imu)
        audio_emb = self.ast(padded)
        hum_emb   = self.tsmixer(humidity)
        return self.fusion(imu_emb, audio_emb, hum_emb)


# ── Data ──────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


def make_loaders(fold):
    train_ds = HARMESDataset(fold["train"], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    val_ds   = HARMESDataset(fold["val"],   DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    test_ds  = HARMESDataset(fold["test"],  DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True)
    return train_loader, val_loader, test_loader, len(train_ds), len(val_ds), len(test_ds)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        labels   = labels.to(device)
        optimizer.zero_grad()
        logits = model(imu, audio, humidity)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        logits   = model(imu, audio, humidity)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return (f1_score(labels, preds, average="macro", zero_division=0),
            accuracy_score(labels, preds))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def best_path(fold_idx):
    return os.path.join(CHECKPOINT_DIR, f"mbt_fold{fold_idx}_best.pt")

def resume_path(fold_idx):
    return os.path.join(CHECKPOINT_DIR, f"mbt_fold{fold_idx}_resume.pt")


def save_resume(fold_idx, epoch, model, optimizer, scheduler,
                best_val_f1, best_val_acc, best_state, no_improve):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_f1":    best_val_f1,
        "best_val_acc":   best_val_acc,
        "best_state":     best_state,
        "no_improve":     no_improve,
    }, resume_path(fold_idx))


def load_resume(fold_idx, model, optimizer, scheduler):
    path = resume_path(fold_idx)
    if not os.path.exists(path):
        return 0, 0.0, 0.0, None, 0   # start_epoch, best_val_f1, best_val_acc, best_state, no_improve

    print(f"  Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"  Resumed at epoch {ckpt['epoch']+1}  "
          f"best_val_f1={ckpt['best_val_f1']:.4f}  no_improve={ckpt['no_improve']}")
    return (ckpt["epoch"], ckpt["best_val_f1"], ckpt["best_val_acc"],
            ckpt["best_state"], ckpt["no_improve"])


# ── Per-fold training ─────────────────────────────────────────────────────────

def train_fold(fold_idx, fold, device):
    print(f"\n{'='*60}")
    print(f"MBT — Fold {fold_idx}")
    print(f"  train={fold['train']}")
    print(f"  val  ={fold['val']}")
    print(f"  test ={fold['test']}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, n_train, n_val, n_test = make_loaders(fold)
    print(f"Train windows: {n_train:,}  |  Val windows: {n_val:,}  |  Test windows: {n_test:,}")

    model     = MBTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # Resume if a checkpoint exists from a previous (timed-out) run
    start_epoch, best_val_f1, best_val_acc, best_state, no_improve = load_resume(
        fold_idx, model, optimizer, scheduler
    )

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_f1, val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  val_F1={val_f1:.4f}  val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(best_state, best_path(fold_idx))
            print(f"  ✓ Best checkpoint (val_F1={val_f1:.4f}) → {best_path(fold_idx)}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (best val_F1={best_val_f1:.4f})")
                break

        save_resume(fold_idx, epoch, model, optimizer, scheduler,
                    best_val_f1, best_val_acc, best_state, no_improve)

    # ── Final unbiased test eval with best checkpoint ──
    if os.path.exists(best_path(fold_idx)):
        model.load_state_dict(torch.load(best_path(fold_idx), map_location=device))
        print(f"\n  Loaded best checkpoint for final test eval.")
    else:
        print(f"\n  WARNING: no best checkpoint saved — evaluating final model state on test.")
    test_f1, test_acc = evaluate(model, test_loader, device)
    print(f"\nFold {fold_idx} FINAL TEST: F1={test_f1:.4f}  acc={test_acc:.4f}  "
          f"(best val_F1={best_val_f1:.4f})")
    return test_f1, test_acc, best_val_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2],
                        help="Which CV fold to train (0, 1, or 2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE/50:.0f}s  |  "
          f"Step: {STEP} samples (50% overlap)")
    print(f"Fold:   {args.fold}\n")

    splits    = get_3fold_splits_with_val(list(range(1, 21)))
    t_start   = time.time()

    test_f1, test_acc, best_val_f1 = train_fold(args.fold, splits[args.fold], device)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"MBT 10s Fold {args.fold} FINAL TEST: F1={test_f1:.4f}  acc={test_acc:.4f}  "
          f"(best val_F1={best_val_f1:.4f})")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Best checkpoint → {best_path(args.fold)}")


if __name__ == "__main__":
    main()
