"""
TinyHAR encoder for IMU data.

Paper: Zhou et al., ISWC 2022 — "TinyHAR: A Lightweight Deep Learning Model
       Designed for Human Activity Recognition"
Repo:  https://github.com/teco-kit/whar-datasets (MIT licensed)

Architecture:
  1. Depthwise separable temporal convolutions — extract local temporal features
     per channel independently, then mix across channels
  2. Temporal self-attention — capture long-range dependencies over time
  3. Global average pooling — collapse time dimension
  4. Linear projection — output fixed-size embedding for fusion

Input:  (B, T, 12) — acc_x/y/z + gyro_x/y/z from both wrists concatenated
Output: (B, embed_dim)
"""

import torch
import torch.nn as nn


class _DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (per channel) followed by pointwise conv (mix channels)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      padding=kernel_size // 2, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TinyHAREncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 12,
        embed_dim: int = 128,
        num_filters: int = 64,
        kernel_size: int = 5,
        num_conv_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stage 1: depthwise separable convolutions along time axis
        layers = [_DepthwiseSeparableConv(in_channels, num_filters, kernel_size, dropout)]
        for _ in range(num_conv_layers - 1):
            layers.append(_DepthwiseSeparableConv(num_filters, num_filters, kernel_size, dropout))
        self.conv_block = nn.Sequential(*layers)

        # Stage 2: temporal self-attention across time steps
        self.temporal_attn = nn.MultiheadAttention(
            num_filters, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(num_filters)
        self.attn_dropout = nn.Dropout(dropout)

        # Stage 3: projection to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(num_filters, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x.permute(0, 2, 1)              # (B, C, T) for Conv1d
        h = self.conv_block(h)               # (B, num_filters, T)
        h = h.permute(0, 2, 1)              # (B, T, num_filters)

        # Temporal self-attention with residual
        attn_out, _ = self.temporal_attn(h, h, h)
        h = self.attn_norm(h + self.attn_dropout(attn_out))  # (B, T, num_filters)

        # Global average pool over time → (B, num_filters)
        h = h.mean(dim=1)

        return self.proj(h)                  # (B, embed_dim)
