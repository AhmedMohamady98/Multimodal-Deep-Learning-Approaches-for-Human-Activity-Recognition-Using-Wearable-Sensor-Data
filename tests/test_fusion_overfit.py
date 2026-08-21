"""
Overfit-on-one-batch tests for fusion modules.

Fusion modules take pre-computed embeddings as input, so these tests
use random embeddings directly — no encoder forward pass needed.
All tests run on CPU in seconds.

Run with:  pytest tests/test_fusion_overfit.py -v -s
"""

import pytest
import torch
import torch.nn as nn
from src.data.synthetic import make_labels, NUM_CLASSES

BATCH = 4
IMU_DIM = 64
AUDIO_DIM = 768
HUM_DIM = 64
STEPS = 100
LR = 1e-3


def _make_embeddings(batch=BATCH):
    return (
        torch.randn(batch, IMU_DIM),
        torch.randn(batch, AUDIO_DIM),
        torch.randn(batch, HUM_DIM),
    )


def _train_fusion(model, imu, audio, hum, labels, steps=STEPS, lr=LR, print_every=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    initial_loss = None
    loss_val = None

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(imu, audio, hum)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        if (step + 1) % print_every == 0:
            print(f"  step {step+1:>3}/{steps}  loss={loss_val:.4f}")

    return initial_loss, loss_val


class TestLateFusionOverfit:
    def test_loss_decreases(self, capsys):
        from src.fusion.late_fusion import LateFusion

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = LateFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)

        with capsys.disabled():
            print(f"\nLateFusion overfit test ({STEPS} steps):")
            initial_loss, final_loss = _train_fusion(model, imu, audio, hum, labels)
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_near_zero(self, capsys):
        from src.fusion.late_fusion import LateFusion

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = LateFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)

        with capsys.disabled():
            print(f"\nLateFusion near-zero test ({STEPS} steps):")
            _, final_loss = _train_fusion(model, imu, audio, hum, labels, print_every=50)
            print(f"  final loss: {final_loss:.4f}")

        assert final_loss < 0.1, f"Final loss {final_loss:.4f} > 0.1"


class TestMBTOverfit:
    def test_loss_decreases(self, capsys):
        from src.fusion.mbt import MultimodalBottleneckTransformer

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        # Small config for fast CPU test
        model = MultimodalBottleneckTransformer(
            IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES,
            model_dim=64, num_bottleneck_tokens=4, num_layers=2, num_heads=4,
        )

        with capsys.disabled():
            print(f"\nMBT overfit test ({STEPS} steps):")
            initial_loss, final_loss = _train_fusion(model, imu, audio, hum, labels)
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_near_zero(self, capsys):
        from src.fusion.mbt import MultimodalBottleneckTransformer

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = MultimodalBottleneckTransformer(
            IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES,
            model_dim=64, num_bottleneck_tokens=4, num_layers=2, num_heads=4,
        )

        with capsys.disabled():
            print(f"\nMBT near-zero test ({STEPS} steps):")
            _, final_loss = _train_fusion(model, imu, audio, hum, labels, print_every=50)
            print(f"  final loss: {final_loss:.4f}")

        assert final_loss < 0.1, f"Final loss {final_loss:.4f} > 0.1"


class TestCMAOverfit:
    def test_loss_decreases(self, capsys):
        from src.fusion.cma import CrossModalAttention

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = CrossModalAttention(
            IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES,
            model_dim=64, num_heads=4, num_layers=2, hidden_dim=64,
        )

        with capsys.disabled():
            print(f"\nCMA overfit test ({STEPS} steps):")
            initial_loss, final_loss = _train_fusion(model, imu, audio, hum, labels)
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_near_zero(self, capsys):
        from src.fusion.cma import CrossModalAttention

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = CrossModalAttention(
            IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES,
            model_dim=64, num_heads=4, num_layers=2, hidden_dim=64,
        )

        with capsys.disabled():
            print(f"\nCMA near-zero test ({STEPS} steps):")
            _, final_loss = _train_fusion(model, imu, audio, hum, labels, print_every=50)
            print(f"  final loss: {final_loss:.4f}")

        assert final_loss < 0.1, f"Final loss {final_loss:.4f} > 0.1"


class TestLMFOverfit:
    # LMF's trilinear form makes optimization harder than other fusion methods —
    # the gradient for each factor depends on the product of the other two.
    # We use rank=32, higher LR, and more steps to reflect real usage.
    LMF_LR = 5e-3
    LMF_STEPS = 200

    def test_loss_decreases(self, capsys):
        from src.fusion.lmf import LowRankMultimodalFusion

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = LowRankMultimodalFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)

        with capsys.disabled():
            print(f"\nLMF overfit test ({self.LMF_STEPS} steps, lr={self.LMF_LR}):")
            initial_loss, final_loss = _train_fusion(
                model, imu, audio, hum, labels,
                steps=self.LMF_STEPS, lr=self.LMF_LR, print_every=50,
            )
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_near_zero(self, capsys):
        from src.fusion.lmf import LowRankMultimodalFusion

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = LowRankMultimodalFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)

        with capsys.disabled():
            print(f"\nLMF near-zero test ({self.LMF_STEPS} steps, lr={self.LMF_LR}):")
            _, final_loss = _train_fusion(
                model, imu, audio, hum, labels,
                steps=self.LMF_STEPS, lr=self.LMF_LR, print_every=100,
            )
            print(f"  final loss: {final_loss:.4f}")

        assert final_loss < 0.1, f"Final loss {final_loss:.4f} > 0.1"


class TestGMFOverfit:
    def test_loss_decreases(self, capsys):
        from src.fusion.gmf import GatedMultimodalFusion

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = GatedMultimodalFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)

        with capsys.disabled():
            print(f"\nGMF overfit test ({STEPS} steps):")
            initial_loss, final_loss = _train_fusion(model, imu, audio, hum, labels)
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_near_zero(self, capsys):
        from src.fusion.gmf import GatedMultimodalFusion

        imu, audio, hum = _make_embeddings()
        labels = make_labels(BATCH)
        model = GatedMultimodalFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)

        with capsys.disabled():
            print(f"\nGMF near-zero test ({STEPS} steps):")
            _, final_loss = _train_fusion(model, imu, audio, hum, labels, print_every=50)
            print(f"  final loss: {final_loss:.4f}")

        assert final_loss < 0.1, f"Final loss {final_loss:.4f} > 0.1"
