"""
2-modality (IMU + Audio, no humidity) ablation training — 10s windows.

Why this exists: the headline 3-modality results (Late Fusion / GMF / CMA)
need an apples-to-apples 2-modality counterpart to quantify the humidity
contribution. The HARMES paper Section 4 expects this comparison.

Architecture (per fold):
    TinyHAR (IMU)   → 128-dim ─┐
    AST (Audio)     → 128-dim ─┴→ 2-modality fusion module → 16 classes
    (no TSMixer, no humidity input)

The 2-mod fusion module classes (LateFusion2Mod / GMF2Mod / CMA2Mod) are
defined here AND mirrored in jobs/evaluate_ablation_2mod_10s.py — keep
both copies in sync if you change layer sizes.

Trains the leakage-free protocol (Option A):
    fold k: train on 18 participants, val on 1, test on the held-out 7

Checkpoints saved to:
    results/models/{method}_2mod_fold{i}_best.pt   (best weights)
    results/models/{method}_2mod_fold{i}_resume.pt (full state)

Usage:
    PYTHONPATH=. python jobs/train_ablation_2mod_10s.py --method late_fusion --fold 0
    PYTHONPATH=. python jobs/train_ablation_2mod_10s.py --method gmf         --fold 1
    PYTHONPATH=. python jobs/train_ablation_2mod_10s.py --method cma         --fold 2

Submit all 9 (3 methods × 3 folds) via the SLURM array in
jobs/train_ablation_2mod_10s.sh.
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
from src.encoders.ast.model import ASTEncoder

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
CHECKPOINT_DIR = "results/models"

WINDOW_SIZE = 500    # 10s × 50 Hz
STEP        = 250    # 50% overlap (matches 3-mod 3-fold CV scripts)
EMBED_DIM   = 128
BATCH_SIZE  = 32
NUM_WORKERS = 2   # lowered from 4: 9-task array hit kernel OOM when 3+ tasks
                  # shared a node with pin_memory=False  # disabled: caused kernel OOM-kill when multiple tasks shared a node + 4 workers each
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10


# ── 2-modality fusion modules (kept in sync with evaluate_ablation_2mod_10s.py) ──

class LateFusion2Mod(nn.Module):
    """Concat(IMU, Audio) → MLP → num_classes."""
    def __init__(self, imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(imu_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, imu_emb, audio_emb):
        return self.classifier(torch.cat([imu_emb, audio_emb], dim=-1))


class GMF2Mod(nn.Module):
    """Gated Multimodal Fusion — 2 modalities (IMU + Audio)."""
    def __init__(self, imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.gate_imu   = nn.Linear(imu_dim,   1)
        self.gate_audio = nn.Linear(audio_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(imu_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, imu_emb, audio_emb):
        g_imu   = torch.sigmoid(self.gate_imu(imu_emb))
        g_audio = torch.sigmoid(self.gate_audio(audio_emb))
        fused   = torch.cat([g_imu * imu_emb, g_audio * audio_emb], dim=-1)
        return self.classifier(fused)


class CMA2Mod(nn.Module):
    """Cross-Modal Attention — 2 modalities (IMU + Audio)."""
    def __init__(self, imu_dim=EMBED_DIM, audio_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_imu_to_audio = nn.MultiheadAttention(
            embed_dim=imu_dim,   num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_audio_to_imu = nn.MultiheadAttention(
            embed_dim=audio_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_imu   = nn.LayerNorm(imu_dim)
        self.norm_audio = nn.LayerNorm(audio_dim)
        self.classifier = nn.Sequential(
            nn.Linear(imu_dim + audio_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, imu_emb, audio_emb):
        imu_q   = imu_emb.unsqueeze(1)
        audio_q = audio_emb.unsqueeze(1)
        imu_ctx,   _ = self.attn_imu_to_audio(imu_q,   audio_q, audio_q)
        audio_ctx, _ = self.attn_audio_to_imu(audio_q, imu_q,   imu_q)
        imu_out   = self.norm_imu(imu_emb     + imu_ctx.squeeze(1))
        audio_out = self.norm_audio(audio_emb + audio_ctx.squeeze(1))
        return self.classifier(torch.cat([imu_out, audio_out], dim=-1))


FUSION_CLASSES_2MOD = {
    "late_fusion": LateFusion2Mod,
    "gmf":         GMF2Mod,
    "cma":         CMA2Mod,
}


# ── Full 2-mod model wrapper ──────────────────────────────────────────────────

class FusionModel2Mod(nn.Module):
    """IMU + Audio only — no humidity/TSMixer pathway."""
    def __init__(self, fusion_module: nn.Module):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.ast     = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)
        self.fusion  = fusion_module

    def forward(self, imu, audio_list):
        device  = imu.device
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a.to(device)
        return self.fusion(self.tinyhar(imu), self.ast(padded))


# ── Data ──────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    # Humidity tensor is ignored — kept in the tuple for dataset compatibility
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


def make_loaders(fold):
    train_ds = HARMESDataset(fold["train"], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    val_ds   = HARMESDataset(fold["val"],   DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    test_ds  = HARMESDataset(fold["test"],  DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              # pin_memory disabled: caused kernel
                              # OOM-kill when 3+ tasks shared a node
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              # pin_memory disabled: caused kernel
                              # OOM-kill when 3+ tasks shared a node
                              pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              # pin_memory disabled: caused kernel
                              # OOM-kill when 3+ tasks shared a node
                              pin_memory=False)
    return train_loader, val_loader, test_loader, len(train_ds), len(val_ds), len(test_ds)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imu, audio, _humidity, labels in loader:
        imu    = imu.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imu, audio)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imu, audio, _humidity, labels in loader:
        imu    = imu.to(device)
        logits = model(imu, audio)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return (f1_score(labels, preds, average="macro", zero_division=0),
            accuracy_score(labels, preds))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def best_path(method, fold_idx):
    return os.path.join(CHECKPOINT_DIR, f"{method}_2mod_fold{fold_idx}_best.pt")

def resume_path(method, fold_idx):
    return os.path.join(CHECKPOINT_DIR, f"{method}_2mod_fold{fold_idx}_resume.pt")


def save_resume(method, fold_idx, epoch, model, optimizer, scheduler,
                best_val_f1, best_val_acc, best_state, no_improve):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_f1":     best_val_f1,
        "best_val_acc":    best_val_acc,
        "best_state":      best_state,
        "no_improve":      no_improve,
    }, resume_path(method, fold_idx))


def load_resume(method, fold_idx, model, optimizer, scheduler):
    path = resume_path(method, fold_idx)
    if not os.path.exists(path):
        return 0, 0.0, 0.0, None, 0

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

def train_fold(method, fold_idx, fold, device):
    print(f"\n{'='*60}")
    print(f"{method} (2-mod, IMU+Audio) — Fold {fold_idx}")
    print(f"  train={fold['train']}")
    print(f"  val  ={fold['val']}")
    print(f"  test ={fold['test']}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, n_train, n_val, n_test = make_loaders(fold)
    print(f"Train windows: {n_train:,}  |  Val windows: {n_val:,}  |  Test windows: {n_test:,}")

    fusion_cls    = FUSION_CLASSES_2MOD[method]
    fusion_module = fusion_cls()
    model         = FusionModel2Mod(fusion_module).to(device)
    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion     = nn.CrossEntropyLoss()

    start_epoch, best_val_f1, best_val_acc, best_state, no_improve = load_resume(
        method, fold_idx, model, optimizer, scheduler
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
            torch.save(best_state, best_path(method, fold_idx))
            print(f"  ✓ Best checkpoint (val_F1={val_f1:.4f}) → {best_path(method, fold_idx)}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (best val_F1={best_val_f1:.4f})")
                break

        save_resume(method, fold_idx, epoch, model, optimizer, scheduler,
                    best_val_f1, best_val_acc, best_state, no_improve)

    if os.path.exists(best_path(method, fold_idx)):
        model.load_state_dict(torch.load(best_path(method, fold_idx), map_location=device))
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
    parser.add_argument("--method", type=str, required=True,
                        choices=sorted(FUSION_CLASSES_2MOD),
                        help="Which fusion variant (late_fusion / gmf / cma)")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2],
                        help="Which CV fold to train (0, 1, or 2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Window: {WINDOW_SIZE} samples = {WINDOW_SIZE/50:.0f}s  |  "
          f"Step: {STEP} samples (50% overlap)")
    print(f"Method: {args.method} (2-mod, IMU+Audio)")
    print(f"Fold:   {args.fold}\n")

    splits  = get_3fold_splits_with_val(list(range(1, 21)))
    t_start = time.time()

    test_f1, test_acc, best_val_f1 = train_fold(args.method, args.fold, splits[args.fold], device)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"{args.method} 2mod 10s Fold {args.fold} FINAL TEST: "
          f"F1={test_f1:.4f}  acc={test_acc:.4f}  (best val_F1={best_val_f1:.4f})")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Best checkpoint → {best_path(args.method, args.fold)}")


if __name__ == "__main__":
    main()
