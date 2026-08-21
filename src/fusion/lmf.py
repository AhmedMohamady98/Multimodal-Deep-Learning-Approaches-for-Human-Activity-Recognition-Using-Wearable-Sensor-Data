"""
Low-Rank Multimodal Fusion (LMF) — Stretch Goal 2.

Models higher-order cross-modal interactions via low-rank tensor
factorization. The full outer product of all modality embeddings would
require O(d1 × d2 × d3 × num_classes) parameters — exponential in the
number of modalities. LMF decomposes this into rank-r factors, one per
modality, reducing parameter count to O(r × (d1 + d2 + d3 + num_classes)).

Reference: Liu et al., ACL 2018 — "Efficient Low-rank Multimodal Fusion
           with Modality-Specific Factors"

Implementation note: modalities are first projected to a shared proj_dim
before the tensor decomposition. This equalizes the scales of very
different embedding sizes (e.g. audio_dim=768 vs humidity_dim=64) and
makes the trilinear optimization landscape well-conditioned.

Fusion formula (for rank r):
  fused_r  = tanh(imu_h · W_imu_r) ⊙ tanh(audio_h · W_audio_r) ⊙ tanh(hum_h · W_hum_r)
  logits   = Σ_r fused_r · W_out_r

where ⊙ is element-wise product, tanh bounds values before the product,
and _h denotes the projected representation with bias appended.

Input:  imu_emb (B, imu_dim), audio_emb (B, audio_dim), humidity_emb (B, humidity_dim)
Output: logits (B, num_classes)
"""

import torch
import torch.nn as nn


class LowRankMultimodalFusion(nn.Module):

    def __init__(
        self,
        imu_dim: int,
        audio_dim: int,
        humidity_dim: int,
        num_classes: int,
        rank: int = 32,
        proj_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rank = rank

        # Project all modalities to a common proj_dim before tensor decomposition.
        # Equalizes scale across very different embedding sizes and makes the
        # trilinear optimization landscape well-conditioned.
        self.imu_proj   = nn.Linear(imu_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.hum_proj   = nn.Linear(humidity_dim, proj_dim)

        # Low-rank factor matrices — +1 on proj_dim for bias row
        self.imu_factor   = nn.Parameter(torch.empty(rank, proj_dim + 1))
        self.audio_factor = nn.Parameter(torch.empty(rank, proj_dim + 1))
        self.hum_factor   = nn.Parameter(torch.empty(rank, proj_dim + 1))
        self.out_factor   = nn.Parameter(torch.empty(rank, num_classes))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for linear in [self.imu_proj, self.audio_proj, self.hum_proj]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        for p in [self.imu_factor, self.audio_factor, self.hum_factor, self.out_factor]:
            nn.init.xavier_uniform_(p)

    def forward(
        self,
        imu_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        humidity_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = imu_emb.shape[0]
        device = imu_emb.device

        # Project to common dimension
        h_imu   = self.dropout(torch.relu(self.imu_proj(imu_emb)))
        h_audio = self.dropout(torch.relu(self.audio_proj(audio_emb)))
        h_hum   = self.dropout(torch.relu(self.hum_proj(humidity_emb)))

        # Append bias term → (B, proj_dim+1)
        ones    = torch.ones(B, 1, device=device)
        h_imu   = torch.cat([h_imu,   ones], dim=-1)
        h_audio = torch.cat([h_audio, ones], dim=-1)
        h_hum   = torch.cat([h_hum,   ones], dim=-1)

        # Project through factor matrices, apply tanh to bound before product
        imu_f   = torch.tanh(h_imu   @ self.imu_factor.T)   # (B, rank)
        audio_f = torch.tanh(h_audio @ self.audio_factor.T) # (B, rank)
        hum_f   = torch.tanh(h_hum   @ self.hum_factor.T)   # (B, rank)

        # Element-wise product — captures 3-way cross-modal interactions
        fused = imu_f * audio_f * hum_f                     # (B, rank)

        # Project to output: (B, rank) @ (rank, num_classes)
        return fused @ self.out_factor                       # (B, num_classes)
