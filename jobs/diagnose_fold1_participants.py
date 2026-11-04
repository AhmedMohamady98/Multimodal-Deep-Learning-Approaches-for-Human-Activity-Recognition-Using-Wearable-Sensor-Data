"""
Per-participant breakdown for fold 1 (test participants 8–14).

Loads the fold 1 checkpoint for each model and runs inference on each
participant individually to identify outliers (e.g. left-handed participants
whose IMU signal is mirrored relative to training data).

Usage:
    PYTHONPATH=. python jobs/diagnose_fold1_participants.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.data.dataset import HARMESDataset, NUM_CLASSES, ID_TO_ACTIVITY
from src.encoders.tinyhar.model import TinyHAREncoder
from src.encoders.tsmixer.model import TSMixerEncoder
from src.encoders.ast.model import ASTEncoder

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
MODELS_DIR = "results/models"
WINDOW_SIZE = 500
STEP        = 250

FOLD1_PARTICIPANTS = [8, 9, 10, 11, 12, 13, 14]

SHORT_NAMES = [
    "Wash hands", "Floor clean", "Window clean", "Vacuum",
    "Brush teeth", "Wash dishes", "Put away dishes", "Make tea",
    "Cut veg", "Cream hands", "Drink", "Water plants",
    "Clean table", "Disinfect", "Clean dishwasher",
]


class UnimodalModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        return self.classifier(self.encoder(x))


class ASTModel(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.encoder    = ASTEncoder(embed_dim=embed_dim, freeze_backbone=True)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, audio_list):
        max_len = max(a.shape[0] for a in audio_list)
        padded  = torch.zeros(len(audio_list), max_len, device=audio_list[0].device)
        for i, a in enumerate(audio_list):
            padded[i, :a.shape[0]] = a
        return self.classifier(self.encoder(padded))


def collate_fn(batch):
    imus, audios, hums, labels = zip(*batch)
    return torch.stack(imus), list(audios), torch.stack(hums), torch.stack(labels)


@torch.no_grad()
def run_inference(model, loader, device, modality):
    model.eval()
    all_preds, all_labels = [], []
    for imu, audio, humidity, labels in loader:
        imu      = imu.to(device)
        humidity = humidity.to(device)
        if modality == "imu":
            logits = model(imu)
        elif modality == "humidity":
            logits = model(humidity)
        elif modality == "audio":
            logits = model([a.to(device) for a in audio])
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def load_model(name, device):
    embed_dim = 128
    fold = 1
    ckpt = os.path.join(MODELS_DIR, f"{name.lower()}_fold{fold}.pt")
    if not os.path.exists(ckpt):
        print(f"  ⚠️  {ckpt} not found"); return None, None

    if name == "TinyHAR":
        model = UnimodalModel(TinyHAREncoder(in_channels=12, embed_dim=embed_dim),
                              embed_dim, NUM_CLASSES)
        modality = "imu"
    elif name == "TSMixer":
        model = UnimodalModel(TSMixerEncoder(seq_len=WINDOW_SIZE, in_channels=1,
                                             embed_dim=embed_dim),
                              embed_dim, NUM_CLASSES)
        modality = "humidity"
    elif name == "AST":
        model = ASTModel(embed_dim, NUM_CLASSES)
        modality = "audio"

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    return model, modality


def evaluate_participant(pid, model, modality, device):
    ds = HARMESDataset([pid], DATA_DIR, window_size=WINDOW_SIZE, step=STEP)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    preds, labels = run_inference(model, loader, device, modality)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    per_class_f1 = f1_score(labels, preds, average=None,
                            labels=list(range(NUM_CLASSES)), zero_division=0)
    return {
        "n_windows":    len(ds),
        "macro_f1":     float(f1),
        "accuracy":     float(acc),
        "per_class_f1": per_class_f1,
        "preds":        preds,
        "labels":       labels,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fold 1 test participants: {FOLD1_PARTICIPANTS}\n")

    for model_name in ["TinyHAR", "AST", "TSMixer"]:
        print("=" * 65)
        print(f"{model_name} — Per-Participant F1 (fold 1 checkpoint)")
        print("=" * 65)

        model, modality = load_model(model_name, device)
        if model is None:
            continue

        results = {}
        for pid in FOLD1_PARTICIPANTS:
            r = evaluate_participant(pid, model, modality, device)
            if r:
                results[pid] = r

        # Summary table
        print(f"  {'P':>3}  {'F1':>7}  {'Acc':>7}  {'Windows':>8}  {'Worst class (F1)'}")
        print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*30}")
        f1_values = []
        for pid, r in results.items():
            worst_idx = int(np.argmin(r["per_class_f1"]))
            worst_f1  = r["per_class_f1"][worst_idx]
            flag = "  ← LOW" if r["macro_f1"] < 0.60 else ""
            print(f"  P{pid:>2}  {r['macro_f1']:>7.4f}  {r['accuracy']:>7.4f}  "
                  f"{r['n_windows']:>8,}  {SHORT_NAMES[worst_idx]} ({worst_f1:.2f}){flag}")
            f1_values.append(r["macro_f1"])

        mean_f1 = np.mean(f1_values)
        std_f1  = np.std(f1_values)
        print(f"\n  Fold 1 aggregate:  F1={mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Min participant:   P{min(results, key=lambda p: results[p]['macro_f1'])} "
              f"(F1={min(r['macro_f1'] for r in results.values()):.4f})")
        print(f"  Max participant:   P{max(results, key=lambda p: results[p]['macro_f1'])} "
              f"(F1={max(r['macro_f1'] for r in results.values()):.4f})")

        # Per-class breakdown for the weakest participant
        weakest_pid = min(results, key=lambda p: results[p]["macro_f1"])
        r = results[weakest_pid]
        print(f"\n  P{weakest_pid} per-class F1 (weakest participant):")
        for i, (name, f1) in enumerate(zip(SHORT_NAMES, r["per_class_f1"])):
            bar = "█" * int(f1 * 20)
            print(f"    {name:<22} {f1:.3f}  {bar}")
        print()


if __name__ == "__main__":
    main()
