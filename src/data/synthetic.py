"""
Synthetic data generator for smoke testing.

Produces correctly-shaped random tensors for each modality so encoders and
fusion modules can be tested locally without the real dataset.

Placeholder sizes (update once EDA confirms sampling rates):
  IMU      — 128 timesteps  (assumes ~50 Hz, ~2.5 s window)
  Audio    — 16000 samples  (16 kHz, 1 s window)
  Humidity — 64 timesteps   (assumes ~1 Hz, ~64 s window)
"""

import torch

from src.data.dataset import NUM_CLASSES   # canonical (includes Null class)


# Placeholder window sizes — swap these out after EDA
IMU_SEQ_LEN = 128       # timesteps
AUDIO_SEQ_LEN = 16000   # raw waveform samples at 16 kHz
HUMIDITY_SEQ_LEN = 64   # timesteps
IMU_CHANNELS = 12       # acc_x/y/z + gyro_x/y/z × 2 wrists


def make_imu_batch(batch_size: int = 4, seq_len: int = IMU_SEQ_LEN) -> torch.Tensor:
    """Returns (B, T, 12) float32 tensor."""
    return torch.randn(batch_size, seq_len, IMU_CHANNELS)


def make_audio_batch(batch_size: int = 4, num_samples: int = AUDIO_SEQ_LEN) -> torch.Tensor:
    """Returns (B, num_samples) float32 tensor — raw waveform."""
    return torch.randn(batch_size, num_samples)


def make_humidity_batch(batch_size: int = 4, seq_len: int = HUMIDITY_SEQ_LEN) -> torch.Tensor:
    """Returns (B, T, 1) float32 tensor."""
    return torch.randn(batch_size, seq_len, 1)


def make_labels(batch_size: int = 4, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Returns (B,) long tensor of random class indices."""
    return torch.randint(0, num_classes, (batch_size,))


def make_full_batch(batch_size: int = 4) -> dict:
    """Returns all modalities + labels in one dict."""
    return {
        "imu": make_imu_batch(batch_size),
        "audio": make_audio_batch(batch_size),
        "humidity": make_humidity_batch(batch_size),
        "labels": make_labels(batch_size),
    }
