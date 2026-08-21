"""
Overfit-on-one-batch tests.

Feeds a single batch of synthetic data to the model and trains for N steps.
A working model must drive loss to near zero — if it can't overfit 1 batch,
something is broken (dead gradients, wrong loss, dimension mismatch).

Run with:  pytest tests/test_overfit.py -v -s
The -s flag shows the loss curve so you can see it actually dropping.
"""

import pytest
import torch
import torch.nn as nn
from src.data.synthetic import (
    make_imu_batch,
    make_humidity_batch,
    make_labels,
    HUMIDITY_SEQ_LEN,
    NUM_CLASSES,
)

BATCH = 4
EMBED_DIM = 64
STEPS = 100
LR = 1e-3


def _train_one_batch(model, x, labels, steps=STEPS, lr=LR, print_every=20):
    """Attaches a linear head, trains for `steps` steps, returns (initial_loss, final_loss)."""
    head = nn.Linear(EMBED_DIM, NUM_CLASSES)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    head.train()

    initial_loss = None
    loss_val = None

    for step in range(steps):
        optimizer.zero_grad()
        emb = model(x)
        logits = head(emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        if (step + 1) % print_every == 0:
            print(f"  step {step+1:>3}/{steps}  loss={loss_val:.4f}")

    return initial_loss, loss_val


class TestTinyHAROverfit:
    def test_loss_decreases(self, capsys):
        from src.encoders.tinyhar.model import TinyHAREncoder

        x = make_imu_batch(BATCH)
        labels = make_labels(BATCH)
        model = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)

        with capsys.disabled():
            print(f"\nTinyHAR overfit test ({STEPS} steps):")
            initial_loss, final_loss = _train_one_batch(model, x, labels)
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}. "
            "Model may not be learning."
        )

    def test_loss_near_zero(self, capsys):
        """With enough steps the model should nearly memorize 1 batch."""
        from src.encoders.tinyhar.model import TinyHAREncoder

        x = make_imu_batch(BATCH)
        labels = make_labels(BATCH)
        model = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)

        with capsys.disabled():
            print(f"\nTinyHAR near-zero test ({STEPS} steps):")
            _, final_loss = _train_one_batch(model, x, labels, steps=STEPS, print_every=25)
            print(f"  final loss: {final_loss:.4f}")

        assert final_loss < 0.1, (
            f"Model could not memorize 1 batch — final loss {final_loss:.4f} > 0.1. "
            "Check architecture or gradients."
        )


class TestASTOverfit:
    """With frozen backbone, only the projection head trains.
    Fixed embeddings + learnable head can still perfectly overfit 1 batch."""

    def test_loss_decreases(self, capsys):
        from src.encoders.ast.model import ASTEncoder
        from src.data.synthetic import make_audio_batch

        x = make_audio_batch(BATCH)
        labels = make_labels(BATCH)
        model = ASTEncoder(embed_dim=EMBED_DIM, freeze_backbone=True)

        with capsys.disabled():
            print(f"\nAST overfit test — frozen backbone ({STEPS} steps):")
            initial_loss, final_loss = _train_one_batch(model, x, labels)
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )


class TestTSMixerOverfit:
    def test_loss_decreases(self, capsys):
        from src.encoders.tsmixer.model import TSMixerEncoder

        x = make_humidity_batch(BATCH)
        labels = make_labels(BATCH)
        model = TSMixerEncoder(seq_len=HUMIDITY_SEQ_LEN, in_channels=1, embed_dim=EMBED_DIM)

        with capsys.disabled():
            print(f"\nTSMixer overfit test ({STEPS} steps):")
            initial_loss, final_loss = _train_one_batch(model, x, labels)
            print(f"  initial loss: {initial_loss:.4f}  →  final loss: {final_loss:.4f}")

        assert final_loss < initial_loss * 0.5
