"""
CLS-Token Transformer — Multimodal Fusion.

A single learnable [CLS] token is prepended to the three modality embeddings,
forming a 4-token sequence. Standard transformer encoder layers process the
full sequence with full self-attention. The [CLS] token aggregates information
from all modalities and is used for classification — the same design used in
BERT and ViT.

This is a natural fit for 3-modality fusion: the attention mechanism is
symmetric, the model has no inductive bias about which modalities to trust,
and the [CLS] token learns to pool across them.

Reference: Devlin et al., NAACL 2019 — "BERT: Pre-training of Deep
           Bidirectional Transformers for Language Understanding"
           Dosovitskiy et al., ICLR 2021 — "An Image is Worth 16×16 Words"

Architecture (one forward pass):
  tokens = [CLS, imu_proj, audio_proj, hum_proj] + pos_emb   (B, 4, D)
  tokens ← TransformerEncoderLayer × num_layers               (B, 4, D)
  cls    ← tokens[:, 0, :]                                    (B, D)
  logits ← LayerNorm → Linear(D → num_classes)                (B, C)

Input:  imu_emb (B, imu_dim), audio_emb (B, audio_dim), humidity_emb (B, humidity_dim)
Output: logits (B, num_classes)
"""

import torch
import torch.nn as nn


class CLSTransformerFusion(nn.Module):

    def __init__(
        self,
        imu_dim: int,
        audio_dim: int,
        humidity_dim: int,
        num_classes: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        # Project each modality to shared model_dim
        self.imu_proj   = nn.Linear(imu_dim,      model_dim)
        self.audio_proj = nn.Linear(audio_dim,     model_dim)
        self.hum_proj   = nn.Linear(humidity_dim,  model_dim)

        # Learnable CLS token and positional embeddings (4 positions)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.pos_emb   = nn.Parameter(torch.randn(1, 4, model_dim) * 0.02)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,   # Pre-LN for stable training
            )
            for _ in range(num_layers)
        ])

        # Classify from CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes),
        )

    def forward(
        self,
        imu_emb:      torch.Tensor,
        audio_emb:    torch.Tensor,
        humidity_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = imu_emb.shape[0]

        # Project modalities → (B, 1, model_dim) each
        imu_tok   = self.imu_proj(imu_emb).unsqueeze(1)
        audio_tok = self.audio_proj(audio_emb).unsqueeze(1)
        hum_tok   = self.hum_proj(humidity_emb).unsqueeze(1)

        # Prepend CLS → (B, 4, model_dim)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, imu_tok, audio_tok, hum_tok], dim=1)
        tokens = tokens + self.pos_emb

        # Transformer layers
        for layer in self.encoder:
            tokens = layer(tokens)

        # Classify from CLS token
        cls_out = tokens[:, 0, :]       # (B, model_dim)
        return self.classifier(cls_out)
