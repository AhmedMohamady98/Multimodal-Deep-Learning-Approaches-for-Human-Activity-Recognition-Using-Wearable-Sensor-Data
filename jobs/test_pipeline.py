"""
Pipeline smoke test on 3 participants.
Verifies: dataset loads, shapes are correct, DataLoader batches work.
Run via: sbatch jobs/test_pipeline.sh
"""

import os

import torch
from torch.utils.data import DataLoader
from src.data.dataset import HARMESDataset, ID_TO_ACTIVITY
from src.data.splits import mini_splits

DATA_DIR = os.environ.get("HARMES_DATA", "data/harmes")
PARTICIPANTS = [1, 2, 3]


def collate_fn(batch):
    """Audio has variable length — stack imu/hum, keep audio as list."""
    imus, audios, hums, labels = zip(*batch)
    return (
        torch.stack(imus),
        list(audios),
        torch.stack(hums),
        torch.stack(labels),
    )


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    splits = mini_splits(PARTICIPANTS)

    for fold_idx, fold in enumerate(splits):
        print(f"{'='*55}")
        print(f"Fold {fold_idx}  train={fold['train']}  test={fold['test']}")

        train_ds = HARMESDataset(fold["train"], DATA_DIR)
        test_ds  = HARMESDataset(fold["test"],  DATA_DIR)

        # Shape check on first sample
        imu, audio, hum, label = train_ds[0]
        print(f"  Sample shapes:")
        print(f"    imu:      {tuple(imu.shape)}")
        print(f"    audio:    {tuple(audio.shape)}")
        print(f"    humidity: {tuple(hum.shape)}")
        print(f"    label:    {label.item()} ({ID_TO_ACTIVITY[label.item()]})")

        # Class distribution
        counts = train_ds.class_counts()
        print(f"  Train class counts:")
        for act, cnt in counts.items():
            bar = "#" * (cnt // 10)
            print(f"    {act:<25} {cnt:4d}  {bar}")

        # DataLoader batch check
        loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                            collate_fn=collate_fn)
        imu_b, audio_b, hum_b, label_b = next(iter(loader))
        print(f"  DataLoader batch:")
        print(f"    imu:      {tuple(imu_b.shape)}")
        print(f"    audio:    list of {len(audio_b)} tensors, "
              f"first shape={tuple(audio_b[0].shape)}")
        print(f"    humidity: {tuple(hum_b.shape)}")
        print(f"    labels:   {tuple(label_b.shape)}")
        print()

    print("=== Pipeline test PASSED ===")


if __name__ == "__main__":
    main()
