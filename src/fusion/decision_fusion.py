"""
Decision Fusion (Probability-Level Fusion).

Each modality encoder gets its own linear classifier head. The three softmax
probability distributions are combined via a learned weighted sum, where the
weights are constrained to sum to 1 via softmax normalization.

Unlike embedding-level fusion (Late Fusion, MBT, CMA), this operates in the
decision space — each modality casts a "vote" (a full class distribution) and
the model learns how much to trust each vote globally. This is principled when
modalities have very different reliability levels, as the weights directly
encode per-modality confidence.

The learned weights are a direct thesis result:
    w = softmax(logit_weights)  →  e.g. [w_imu=0.25, w_audio=0.70, w_hum=0.05]
This confirms which modality the fusion learned to rely on most.

Architecture:
  p_imu   = softmax(Linear(imu_dim   → num_classes)(imu_emb))    (B, C)
  p_audio = softmax(Linear(audio_dim → num_classes)(audio_emb))  (B, C)
  p_hum   = softmax(Linear(hum_dim   → num_classes)(hum_emb))    (B, C)

  w = softmax(logit_weights)                                       (3,)
  p_fused = w[0]*p_imu + w[1]*p_audio + w[2]*p_hum               (B, C)

  return log(p_fused)   ← log-probabilities for nn.NLLLoss

Input:  imu_emb (B, imu_dim), audio_emb (B, audio_dim), humidity_emb (B, humidity_dim)
Output: log-probabilities (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionFusion(nn.Module):

    def __init__(
        self,
        imu_dim: int,
        audio_dim: int,
        humidity_dim: int,
        num_classes: int,
    ):
        super().__init__()

        # Per-modality classifier heads
        self.head_imu   = nn.Linear(imu_dim,      num_classes)
        self.head_audio = nn.Linear(audio_dim,     num_classes)
        self.head_hum   = nn.Linear(humidity_dim,  num_classes)

        # Learned logit weights — softmaxed so they sum to 1
        # Initialised to zeros → uniform weights at start of training
        self.logit_weights = nn.Parameter(torch.zeros(3))

    @property
    def modality_weights(self) -> torch.Tensor:
        """Softmax-normalised weights [w_imu, w_audio, w_hum]. For inspection."""
        return F.softmax(self.logit_weights, dim=0).detach()

    def forward(
        self,
        imu_emb:      torch.Tensor,
        audio_emb:    torch.Tensor,
        humidity_emb: torch.Tensor,
    ) -> torch.Tensor:

        # Each head produces a probability distribution over classes
        p_imu   = F.softmax(self.head_imu(imu_emb),     dim=-1)   # (B, C)
        p_audio = F.softmax(self.head_audio(audio_emb), dim=-1)   # (B, C)
        p_hum   = F.softmax(self.head_hum(humidity_emb), dim=-1)  # (B, C)

        # Weighted sum — weights guaranteed to sum to 1
        w = F.softmax(self.logit_weights, dim=0)                   # (3,)
        p_fused = w[0] * p_imu + w[1] * p_audio + w[2] * p_hum   # (B, C)

        # Return log-probs for nn.NLLLoss; clamp avoids log(0)
        return torch.log(p_fused.clamp(min=1e-8))
