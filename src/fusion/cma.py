"""
Cross-Modal Attention (CMA) — Stretch Goal 1.

Each modality queries information from both other modalities via directional
cross-attention. Unlike MBT (which routes through bottleneck tokens), CMA
allows direct, fine-grained attention between all modality pairs — at the
cost of 6 attention modules instead of shared bottleneck tokens.

Reference: Tsai et al., ACL 2019 — "Multimodal Transformer for Unaligned
           Multimodal Language Sequences" (MulT)

Per-layer update (applied num_layers times):
  h_imu   ← LayerNorm(h_imu   + CrossAttn(Q=imu,   K=audio, V=audio)
                                + CrossAttn(Q=imu,   K=hum,   V=hum))
  h_audio ← LayerNorm(h_audio + CrossAttn(Q=audio, K=imu,   V=imu)
                                + CrossAttn(Q=audio, K=hum,   V=hum))
  h_hum   ← LayerNorm(h_hum   + CrossAttn(Q=hum,   K=imu,   V=imu)
                                + CrossAttn(Q=hum,   K=audio, V=audio))

Classification: concatenate final h_imu, h_audio, h_hum → MLP.

Input:  imu_emb (B, imu_dim), audio_emb (B, audio_dim), humidity_emb (B, humidity_dim)
Output: logits (B, num_classes)
"""

import torch
import torch.nn as nn


class _CMALayer(nn.Module):
    """One layer of bidirectional cross-modal attention for 3 modalities."""

    def __init__(self, model_dim: int, num_heads: int,
                 attn_dropout: float = 0.1):
        super().__init__()
        # attn_dropout kept low (0.1): with seq_len=1, the attention weight is a
        # single scalar (always 1.0 after softmax). Higher dropout would zero it
        # N% of steps, completely blocking cross-modal exchange for that pair.
        attn_kwargs = dict(embed_dim=model_dim, num_heads=num_heads,
                           dropout=attn_dropout, batch_first=True)

        # 6 directional cross-attention modules (all pairs, both directions)
        self.ca_imu_audio   = nn.MultiheadAttention(**attn_kwargs)
        self.ca_imu_hum     = nn.MultiheadAttention(**attn_kwargs)
        self.ca_audio_imu   = nn.MultiheadAttention(**attn_kwargs)
        self.ca_audio_hum   = nn.MultiheadAttention(**attn_kwargs)
        self.ca_hum_imu     = nn.MultiheadAttention(**attn_kwargs)
        self.ca_hum_audio   = nn.MultiheadAttention(**attn_kwargs)

        self.norm_imu   = nn.LayerNorm(model_dim)
        self.norm_audio = nn.LayerNorm(model_dim)
        self.norm_hum   = nn.LayerNorm(model_dim)

    def forward(
        self,
        h_imu: torch.Tensor,
        h_audio: torch.Tensor,
        h_hum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Each modality queries from both other modalities, residual sum
        imu_new = self.norm_imu(
            h_imu
            + self.ca_imu_audio(h_imu,   h_audio, h_audio)[0]
            + self.ca_imu_hum(h_imu,     h_hum,   h_hum)[0]
        )
        audio_new = self.norm_audio(
            h_audio
            + self.ca_audio_imu(h_audio, h_imu,   h_imu)[0]
            + self.ca_audio_hum(h_audio, h_hum,   h_hum)[0]
        )
        hum_new = self.norm_hum(
            h_hum
            + self.ca_hum_imu(h_hum,     h_imu,   h_imu)[0]
            + self.ca_hum_audio(h_hum,   h_audio, h_audio)[0]
        )
        return imu_new, audio_new, hum_new


class CrossModalAttention(nn.Module):

    def __init__(
        self,
        imu_dim: int,
        audio_dim: int,
        humidity_dim: int,
        num_classes: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        # Project each modality to shared model_dim
        self.imu_proj   = nn.Linear(imu_dim, model_dim)
        self.audio_proj = nn.Linear(audio_dim, model_dim)
        self.hum_proj   = nn.Linear(humidity_dim, model_dim)

        # Stack of CMA layers — attention dropout fixed at 0.1 regardless of
        # the classifier dropout value passed in
        self.layers = nn.ModuleList([
            _CMALayer(model_dim, num_heads, attn_dropout=0.1)
            for _ in range(num_layers)
        ])

        # Classify from concatenated final representations
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 3, hidden_dim),
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
        # Project and add sequence dimension → (B, 1, model_dim)
        h_imu   = self.imu_proj(imu_emb).unsqueeze(1)
        h_audio = self.audio_proj(audio_emb).unsqueeze(1)
        h_hum   = self.hum_proj(humidity_emb).unsqueeze(1)

        for layer in self.layers:
            h_imu, h_audio, h_hum = layer(h_imu, h_audio, h_hum)

        # Squeeze sequence dim, concatenate all modalities
        fused = torch.cat([
            h_imu.squeeze(1),
            h_audio.squeeze(1),
            h_hum.squeeze(1),
        ], dim=-1)  # (B, 3 * model_dim)

        return self.classifier(fused)
