"""
GMF — Leave-One-Participant-Out (LOPO) training.

Evaluation protocol to match the HARMES paper baseline:
  - 20 folds, each leaving out one participant as the test set
  - 10-second windows, NO overlap (step = window_size = 500 samples)
  - GMF architecture identical to the 3-fold experiments

Results saved to results/lopo/models/
  lopo_p{XX}_best.pt        — best model for participant XX (zero-padded, e.g. lopo_p01_best.pt)
  lopo_p{XX}_resume.pt      — resume checkpoint

Usage:
    PYTHONPATH=. python jobs/train_gmf_lopo.py --participant 1
    ...
    PYTHONPATH=. python jobs/train_gmf_lopo.py --participant 20
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
from src.data.splits import get_lopo_split_with_val
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.gmf import GatedMultimodalFusion

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
CHECKPOINT_DIR = "results/lopo/models"

WINDOW_SIZE = 500   # 10s × 50 Hz
STEP        = 500   # NO overlap — matches HARMES paper protocol
EMBED_DIM   = 128
BATCH_SIZE  = 32
NUM_WORKERS = 4
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10

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


# ── Data ──────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


def make_loaders(test_participant):
    split = get_lopo_split_with_val(test_participant, ALL_PARTICIPANTS)
    train_ds = HARMESDataset(split["train"], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    val_ds   = HARMESDataset(split["val"],   DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    test_ds  = HARMESDataset(split["test"],  DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader, split, len(train_ds), len(val_ds), len(test_ds)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        labels   = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imu, audio, humidity), labels)
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


# ── Checkpointing ─────────────────────────────────────────────────────────────

def best_path(p):   return os.path.join(CHECKPOINT_DIR, f"lopo_p{p:02d}_best.pt")
def resume_path(p): return os.path.join(CHECKPOINT_DIR, f"lopo_p{p:02d}_resume.pt")


def save_resume(p, epoch, model, optimizer, scheduler, best_val_f1, best_val_acc, best_state, no_improve):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "best_val_f1": best_val_f1, "best_val_acc": best_val_acc,
        "best_state": best_state, "no_improve": no_improve,
    }, resume_path(p))


def load_resume(p, model, optimizer, scheduler):
    path = resume_path(p)
    if not os.path.exists(path):
        return 0, 0.0, 0.0, None, 0
    print(f"  Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"  Resumed at epoch {ckpt['epoch']+1}  best_val_f1={ckpt['best_val_f1']:.4f}")
    return ckpt["epoch"], ckpt["best_val_f1"], ckpt["best_val_acc"], ckpt["best_state"], ckpt["no_improve"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=int, required=True, choices=ALL_PARTICIPANTS,
                        help="Participant held out as test set (1–20)")
    args = parser.parse_args()
    p = args.participant

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"LOPO — leaving out participant P{p:02d}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE/50:.0f}s  |  Step: {STEP} (no overlap)")

    train_loader, val_loader, test_loader, split, n_train, n_val, n_test = make_loaders(p)
    print(f"  train={split['train']}")
    print(f"  val  ={split['val']}")
    print(f"  test ={split['test']}")
    print(f"Train windows: {n_train:,}  |  Val windows: {n_val:,}  |  Test windows: {n_test:,}\n")

    model     = GMFModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_val_f1, best_val_acc, best_state, no_improve = load_resume(
        p, model, optimizer, scheduler
    )

    t_start = time.time()
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_f1, val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  val_F1={val_f1:.4f}  val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_val_acc = val_f1, val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(best_state, best_path(p))
            print(f"  ✓ Best checkpoint (val_F1={val_f1:.4f}) → {best_path(p)}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (best val_F1={best_val_f1:.4f})")
                break

        save_resume(p, epoch, model, optimizer, scheduler,
                    best_val_f1, best_val_acc, best_state, no_improve)

    # ── Final unbiased test eval with best checkpoint ──
    if os.path.exists(best_path(p)):
        model.load_state_dict(torch.load(best_path(p), map_location=device))
        print(f"\n  Loaded best checkpoint for final test eval.")
    else:
        print(f"\n  WARNING: no best checkpoint saved — evaluating final model state on test.")
    test_f1, test_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - t_start
    print(f"\nP{p:02d} FINAL TEST: F1={test_f1:.4f}  acc={test_acc:.4f}  "
          f"(best val_F1={best_val_f1:.4f})  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
