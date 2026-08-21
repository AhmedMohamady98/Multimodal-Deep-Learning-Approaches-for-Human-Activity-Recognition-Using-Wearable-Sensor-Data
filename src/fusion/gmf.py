"""
Gated Multimodal Fusion (GMF) — Baseline 2.

Each modality is transformed to a shared hidden space, then weighted by a
learned sigmoid gate computed from all modalities concatenated. This lets
the model suppress uninformative modalities per sample — e.g. the humidity
gate should be near zero for non-water activities.

Reference: Arevalo et al., ICLR Workshop 2017 — "Gated Multimodal Units
           for Information Fusion" (GMU = GMF, same mechanism, extended
           here from 2 to 3 modalities)

Input:  imu_emb (B, imu_dim), audio_emb (B, audio_dim), humidity_emb (B, humidity_dim)
Output: logits (B, num_classes)
"""

import torch
import torch.nn as nn


class GatedMultimodalFusion(nn.Module):

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
        total_dim = imu_dim + audio_dim + humidity_dim

        # Project each modality to a shared hidden space
        self.imu_proj    = nn.Linear(imu_dim, hidden_dim)
        self.audio_proj  = nn.Linear(audio_dim, hidden_dim)
        self.hum_proj    = nn.Linear(humidity_dim, hidden_dim)

        # Gate per modality — conditioned on all modalities concatenated
        # Output is hidden_dim so the gate is element-wise over the hidden space
        self.imu_gate    = nn.Linear(total_dim, hidden_dim)
        self.audio_gate  = nn.Linear(total_dim, hidden_dim)
        self.hum_gate    = nn.Linear(total_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        imu_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        humidity_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Transform each modality
        h_imu   = torch.tanh(self.imu_proj(imu_emb))
        h_audio = torch.tanh(self.audio_proj(audio_emb))
        h_hum   = torch.tanh(self.hum_proj(humidity_emb))

        # Gates are conditioned on all modalities so each gate can react
        # to the presence or absence of signal in the other modalities
        all_emb = torch.cat([imu_emb, audio_emb, humidity_emb], dim=-1)
        z_imu   = torch.sigmoid(self.imu_gate(all_emb))
        z_audio = torch.sigmoid(self.audio_gate(all_emb))
        z_hum   = torch.sigmoid(self.hum_gate(all_emb))

        # Gated sum — each modality contributes proportionally to its gate
        fused = z_imu * h_imu + z_audio * h_audio + z_hum * h_hum  # (B, hidden_dim)

        return self.classifier(fused)
