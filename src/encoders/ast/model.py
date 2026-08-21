"""
AST encoder for audio data.

Paper: Gong et al., Interspeech 2021 — "AST: Audio Spectrogram Transformer"
Model: MIT/ast-finetuned-audioset-10-10-0.4593 (HuggingFace)

Architecture:
  - ASTFeatureExtractor converts raw waveform → log-Mel spectrogram patches
  - ASTModel (ViT-based) processes patches, outputs CLS token embedding
  - Linear projection → fixed-size embedding for fusion

Strategy:
  Phase 1 — freeze backbone, train projection head only (fast, stable)
  Phase 2 — unfreeze backbone for full fine-tuning (once unimodal baseline works)

Input:  (B, num_samples) — raw waveform at 16 kHz
Output: (B, embed_dim)
"""

import torch
import torch.nn as nn
import torchaudio.functional as F_audio
from transformers import ASTModel, ASTFeatureExtractor

HF_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLING_RATE = 16000       # AST / AudioSet expected rate
DATASET_SR    = 44_100      # rate at which HARMES stores audio
AST_HIDDEN_DIM = 768  # ViT-Base hidden size


class ASTEncoder(nn.Module):

    def __init__(self, embed_dim: int = 768, freeze_backbone: bool = True):
        super().__init__()

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(HF_MODEL_ID)
        self.ast = ASTModel.from_pretrained(HF_MODEL_ID)

        if freeze_backbone:
            for param in self.ast.parameters():
                param.requires_grad = False

        # Project CLS embedding → embed_dim (identity if dims match)
        self.proj = (
            nn.Sequential(
                nn.Linear(AST_HIDDEN_DIM, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            if embed_dim != AST_HIDDEN_DIM
            else nn.Identity()
        )

    def unfreeze_backbone(self):
        """Call this to switch from Phase 1 to Phase 2 fine-tuning."""
        for param in self.ast.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_samples) raw waveform on any device
        # ASTFeatureExtractor runs on CPU with numpy — move back to device after
        device = x.device

        # Frozen backbone should stay deterministic even when model.train() is called
        if not next(self.ast.parameters()).requires_grad:
            self.ast.eval()

        # Dataset audio is 44.1 kHz; resample to 16 kHz before feature extraction.
        # Without this the mel spectrogram frames are computed at wrong hop size and
        # 8s/10s windows get truncated to only ~3.7 s of actual audio.
        waveforms = [
            F_audio.resample(x[i].cpu().float(), orig_freq=DATASET_SR, new_freq=SAMPLING_RATE).numpy()
            for i in range(x.shape[0])
        ]

        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)

        outputs = self.ast(input_values=input_values)

        # AST uses two prefix tokens (CLS + distillation, like DeiT)
        # average both for a richer embedding
        cls_emb = outputs.last_hidden_state[:, :2, :].mean(dim=1)  # (B, 768)

        return self.proj(cls_emb)  # (B, embed_dim)
