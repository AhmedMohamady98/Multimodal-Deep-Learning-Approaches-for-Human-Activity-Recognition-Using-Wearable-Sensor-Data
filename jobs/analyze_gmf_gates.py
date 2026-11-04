"""
GMF gate-activation analysis.

The thesis claims in three places that GMF's gates make the fused model
interpretable, because each gate reports how much of its modality the model
admitted for a given window. This script actually reads those gates.

For every test window in every fold it records the mean activation of the
three gate vectors, alongside the true and predicted label, so the question
"which modality does the model rely on for this activity?" can be answered
from the trained weights rather than inferred from ablation.

Method. The three gates are nn.Linear(384 -> 256) layers whose outputs pass
through a sigmoid in GatedMultimodalFusion.forward. Forward hooks capture the
pre-sigmoid output; the sigmoid is applied here, and the 256 dimensions are
averaged into one scalar per modality per window. That scalar is the fraction
of the modality's projected representation that survives gating, so 0 means
the stream was ignored for that window and 1 means it passed through intact.

No retraining and no change to the model: the checkpoints are loaded exactly
as the evaluation script loads them.

Output:
    results/gates/gmf_gate_activations.csv
        fold, true_label, pred_label, g_imu, g_audio, g_hum   (one row/window)

Usage:
    PYTHONPATH=. python jobs/analyze_gmf_gates.py
"""

import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.data.splits import get_3fold_splits
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder
from src.fusion.gmf import GatedMultimodalFusion

# ── Config (mirrors jobs/evaluate_gmf_10s.py) ────────────────────────────────

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR  = "results/models"
OUT_DIR     = "results/gates"
OUT_CSV     = os.path.join(OUT_DIR, "gmf_gate_activations.csv")

WINDOW_SIZE = 500
STEP        = 250
EMBED_DIM   = 128
BATCH_SIZE  = 32


class GMFModel(nn.Module):
    """Identical to the class in evaluate_gmf_10s.py so state_dict loads clean."""

    def __init__(self):
        super().__init__()
        self.tinyhar = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.tsmixer = TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1,
                                      embed_dim=EMBED_DIM)
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
        return self.fusion(self.tinyhar(imu), self.ast(padded),
                           self.tsmixer(humidity))


def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


def attach_gate_hooks(model, store):
    """Capture the pre-sigmoid output of each gate layer.

    GatedMultimodalFusion applies torch.sigmoid to these Linear outputs, so the
    sigmoid is reapplied here to recover the gate values the model actually
    used. Hooking rather than re-deriving keeps this honest: whatever the
    forward pass computed is what gets recorded.
    """
    handles = []
    for name, layer in [("imu",   model.fusion.imu_gate),
                        ("audio", model.fusion.audio_gate),
                        ("hum",   model.fusion.hum_gate)]:
        def make(n):
            def hook(_module, _inp, out):
                store[n] = torch.sigmoid(out).mean(dim=-1).detach().cpu()
            return hook
        handles.append(layer.register_forward_hook(make(name)))
    return handles


@torch.no_grad()
def gates_for_fold(model, loader, device, fold_idx, writer):
    model.eval()
    store = {}
    handles = attach_gate_hooks(model, store)
    all_preds, all_labels = [], []
    try:
        for imu, audio, humidity, labels in loader:
            imu      = imu.to(device)
            humidity = humidity.to(device)
            logits   = model(imu, audio, humidity)
            preds    = logits.argmax(dim=-1).cpu().numpy()
            labs     = labels.numpy()
            g_imu    = store["imu"].numpy()
            g_audio  = store["audio"].numpy()
            g_hum    = store["hum"].numpy()
            for i in range(len(labs)):
                writer.writerow([fold_idx, int(labs[i]), int(preds[i]),
                                 f"{g_imu[i]:.6f}", f"{g_audio[i]:.6f}",
                                 f"{g_hum[i]:.6f}"])
            all_preds.append(preds)
            all_labels.append(labs)
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits  = get_3fold_splits(list(range(1, 21)))
    total   = 0
    fold_f1 = []
    t0      = time.time()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "true_label", "pred_label",
                         "g_imu", "g_audio", "g_hum"])

        for fold_idx, fold in enumerate(splits):
            ckpt = os.path.join(MODELS_DIR, f"gmf_fold{fold_idx}_best.pt")
            if not os.path.exists(ckpt):
                print(f"  Missing {ckpt} — skipping fold {fold_idx}")
                continue

            model = GMFModel().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))

            test_ds = HARMESDataset(fold["test"], DATA_DIR,
                                    window_size=WINDOW_SIZE, step=STEP)
            loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4,
                                 pin_memory=True)

            preds, labels = gates_for_fold(model, loader, device,
                                           fold_idx, writer)
            total += len(labels)

            # Sanity check on the frozen backbone. The AST weights had to be
            # re-fetched after the workspace holding the HF cache expired, so
            # reproducing the published per-fold macro F1 confirms the restored
            # backbone is the same one that produced the reported results.
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            fold_f1.append(f1)
            print(f"  fold {fold_idx}: {len(labels):,} windows   "
                  f"macro F1 = {f1:.4f}   (test participants {fold['test']})")

            del model
            torch.cuda.empty_cache()

    print(f"\nWrote {total:,} rows to {OUT_CSV} in {time.time() - t0:.0f}s")

    if fold_f1:
        mean, std = float(np.mean(fold_f1)), float(np.std(fold_f1))
        print(f"\nMacro F1 across folds: {mean:.4f} +/- {std:.4f}")
        print(f"Published GMF value: 0.827")
        delta = abs(mean - 0.827)
        verdict = ("MATCH — restored AST backbone reproduces the published result"
                   if delta < 0.002 else
                   f"MISMATCH by {delta:.4f} — investigate before using these gates")
        print(f"  -> {verdict}")

    # Console sanity check: mean gate per modality, and the two extremes.
    import collections
    sums = collections.defaultdict(lambda: np.zeros(3))
    cnts = collections.Counter()
    with open(OUT_CSV) as f:
        for row in csv.DictReader(f):
            k = int(row["true_label"])
            sums[k] += np.array([float(row["g_imu"]), float(row["g_audio"]),
                                 float(row["g_hum"])])
            cnts[k] += 1
    print(f"\n{'activity':26}{'IMU':>8}{'audio':>8}{'humidity':>10}{'n':>8}")
    for k in range(NUM_CLASSES):
        if cnts[k]:
            m = sums[k] / cnts[k]
            print(f"{ID_TO_ACTIVITY[k][:24]:26}{m[0]:>8.3f}{m[1]:>8.3f}"
                  f"{m[2]:>10.3f}{cnts[k]:>8,}")


if __name__ == "__main__":
    main()
