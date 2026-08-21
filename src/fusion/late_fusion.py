"""
Late Fusion — Baseline 1.

Concatenates embeddings from all three encoders and passes through a shared
MLP classifier. No cross-modal interaction — each encoder operates fully
independently and modalities are only combined at the decision level.

Reference: standard baseline, no specific paper needed.

Input:  imu_emb (B, imu_dim), audio_emb (B, audio_dim), humidity_emb (B, humidity_dim)
Output: logits (B, num_classes)
"""

import torch
import torch.nn as nn


class LateFusion(nn.Module):

    def __init__(
        self,
        imu_dim: int,
        audio_dim: int,
        humidity_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        fused_dim = imu_dim + audio_dim + humidity_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        imu_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        humidity_emb: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([imu_emb, audio_emb, humidity_emb], dim=-1)  # (B, fused_dim)
        return self.classifier(fused)                                    # (B, num_classes)
