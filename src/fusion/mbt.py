"""
Multimodal Bottleneck Transformer (MBT) — Main Contribution.

A small set of shared learnable bottleneck tokens mediate cross-modal
information exchange between the three encoder streams. At each layer,
each modality attends to its own token AND the bottleneck tokens. The
bottleneck tokens from all three streams are then averaged before the
next layer. This avoids the O((N1+N2+N3)^2) cost of full cross-attention
while still enabling cross-modal interaction.

Reference: Nagrani et al., NeurIPS 2021 — "Attention Bottlenecks for
           Multimodal Fusion"

Layer structure (repeated num_layers times):
  imu_stream   = [h_imu,   bottleneck]  → TransformerLayer → h_imu',   bn_imu
  audio_stream = [h_audio, bottleneck]  → TransformerLayer → h_audio', bn_audio
  hum_stream   = [h_hum,   bottleneck]  → TransformerLayer → h_hum',   bn_hum
  bottleneck   = mean(bn_imu, bn_audio, bn_hum)   ← shared for next layer

Classification: mean-pool final bottleneck tokens → linear head.

Input:  imu_emb (B, imu_dim), audio_emb (B, audio_dim), humidity_emb (B, humidity_dim)
Output: logits (B, num_classes)
"""

import torch
import torch.nn as nn


class MultimodalBottleneckTransformer(nn.Module):

    def __init__(
        self,
        imu_dim: int,
        audio_dim: int,
        humidity_dim: int,
        num_classes: int,
        model_dim: int = 256,
        num_bottleneck_tokens: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        # Project each modality embedding to shared model_dim
        self.imu_proj   = nn.Linear(imu_dim, model_dim)
        self.audio_proj = nn.Linear(audio_dim, model_dim)
        self.hum_proj   = nn.Linear(humidity_dim, model_dim)

        # Learnable bottleneck tokens — shared across all modalities
        self.bottleneck = nn.Parameter(
            torch.randn(1, num_bottleneck_tokens, model_dim) * 0.02
        )

        # Independent transformer layers per modality per depth level
        # Each processes [modality_token (1), bottleneck_tokens (N_b)]
        layer_kwargs = dict(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.imu_layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(**layer_kwargs) for _ in range(num_layers)
        ])
        self.audio_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(**layer_kwargs) for _ in range(num_layers)
        ])
        self.hum_layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(**layer_kwargs) for _ in range(num_layers)
        ])

        self.num_bottleneck_tokens = num_bottleneck_tokens

        # Classify from mean-pooled final bottleneck tokens
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes),
        )

    def forward(
        self,
        imu_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        humidity_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = imu_emb.shape[0]

        # Project to model_dim, add sequence dimension → (B, 1, model_dim)
        h_imu   = self.imu_proj(imu_emb).unsqueeze(1)
        h_audio = self.audio_proj(audio_emb).unsqueeze(1)
        h_hum   = self.hum_proj(humidity_emb).unsqueeze(1)

        # Expand bottleneck tokens to batch size → (B, N_b, model_dim)
        bn = self.bottleneck.expand(B, -1, -1)

        for imu_layer, audio_layer, hum_layer in zip(
            self.imu_layers, self.audio_layers, self.hum_layers
        ):
            # Build each stream: [modality_token, bottleneck]
            imu_stream   = torch.cat([h_imu,   bn], dim=1)  # (B, 1+N_b, model_dim)
            audio_stream = torch.cat([h_audio, bn], dim=1)
            hum_stream   = torch.cat([h_hum,   bn], dim=1)

            # Apply transformer layer to each stream independently
            imu_out   = imu_layer(imu_stream)
            audio_out = audio_layer(audio_stream)
            hum_out   = hum_layer(hum_stream)

            # Split updated modality tokens and bottleneck tokens
            h_imu,   bn_imu   = imu_out[:, :1],   imu_out[:, 1:]
            h_audio, bn_audio = audio_out[:, :1],  audio_out[:, 1:]
            h_hum,   bn_hum   = hum_out[:, :1],    hum_out[:, 1:]

            # Average bottleneck tokens across modalities — the shared state
            bn = (bn_imu + bn_audio + bn_hum) / 3.0

        # Mean-pool final bottleneck tokens → classify
        out = bn.mean(dim=1)        # (B, model_dim)
        return self.classifier(out)
