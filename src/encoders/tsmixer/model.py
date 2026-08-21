"""
TSMixer encoder for humidity time series.

Paper: Chen et al., TMLR 2023 — "TSMixer: An All-MLP Architecture for
       Time Series Forecasting" (arXiv 2303.06053)
Repo ref: https://github.com/ditschuk/pytorch-tsmixer (MIT licensed)

Architecture — alternating MLP-Mixer blocks:
  1. Input projection: (B, T, 1) → (B, T, hidden_dim)
  2. N × TSMixerBlock:
       TimeMixing   — linear across T for each feature independently (residual)
       FeatureMixing — 2-layer MLP across feature dim for each timestep (residual)
  3. Global average pool over T → (B, hidden_dim)
  4. Linear projection → (B, embed_dim)

Input:  (B, T, 1) — single-channel humidity signal
Output: (B, embed_dim)
"""

import torch
import torch.nn as nn


class _TimeMixing(nn.Module):
    """Mix information across the time dimension, independently per feature."""

    def __init__(self, seq_len: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.linear = nn.Linear(seq_len, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x.permute(0, 2, 1)      # (B, C, T)
        h = self.norm(h)
        h = torch.relu(self.linear(h))
        h = self.dropout(h)
        h = h.permute(0, 2, 1)      # (B, T, C)
        return x + h                 # residual


class _FeatureMixing(nn.Module):
    """2-layer MLP across the feature dimension at each timestep."""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, hidden_dim)
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h                 # residual


class _TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.time_mixing = _TimeMixing(seq_len, dropout)
        self.feature_mixing = _FeatureMixing(hidden_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x


class TSMixerEncoder(nn.Module):

    def __init__(
        self,
        seq_len: int = 64,
        in_channels: int = 1,
        hidden_dim: int = 32,
        ff_dim: int = 64,
        num_blocks: int = 4,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project single humidity channel to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Stack of TSMixer blocks
        self.blocks = nn.ModuleList([
            _TSMixerBlock(seq_len, hidden_dim, ff_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output: pool over time then project to embed_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        h = self.input_proj(x)       # (B, T, hidden_dim)
        for block in self.blocks:
            h = block(h)             # (B, T, hidden_dim)
        h = h.mean(dim=1)            # (B, hidden_dim) — global avg pool over T
        return self.proj(h)          # (B, embed_dim)
