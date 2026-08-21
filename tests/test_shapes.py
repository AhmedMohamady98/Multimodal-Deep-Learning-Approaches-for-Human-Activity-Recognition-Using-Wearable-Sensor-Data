"""
Smoke tests — forward pass shape checks for all encoders and fusion modules.

Run with:  pytest tests/test_shapes.py -v
All tests run on CPU in seconds. No real data needed.
"""

import pytest
import torch
from src.data.synthetic import (
    make_imu_batch,
    make_audio_batch,
    make_humidity_batch,
    IMU_SEQ_LEN,
    AUDIO_SEQ_LEN,
    HUMIDITY_SEQ_LEN,
    NUM_CLASSES,
)

BATCH = 4
EMBED_DIM = 64  # small for fast tests


# ---------------------------------------------------------------------------
# Encoder smoke tests
# ---------------------------------------------------------------------------

class TestTinyHAREncoder:
    def setup_method(self):
        from src.encoders.tinyhar.model import TinyHAREncoder
        self.model = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        self.model.eval()

    def test_output_shape(self):
        x = make_imu_batch(BATCH)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (BATCH, EMBED_DIM), f"Expected ({BATCH}, {EMBED_DIM}), got {out.shape}"

    def test_no_nan(self):
        x = make_imu_batch(BATCH)
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_different_batch_sizes(self):
        from src.encoders.tinyhar.model import TinyHAREncoder
        model = TinyHAREncoder(in_channels=12, embed_dim=EMBED_DIM)
        model.eval()
        for b in [1, 2, 8]:
            x = make_imu_batch(b)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (b, EMBED_DIM)


class TestASTEncoder:
    def setup_method(self):
        from src.encoders.ast.model import ASTEncoder
        self.model = ASTEncoder(embed_dim=768, freeze_backbone=True)
        self.model.eval()

    def test_output_shape(self):
        x = make_audio_batch(BATCH)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (BATCH, 768), f"Expected ({BATCH}, 768), got {out.shape}"

    def test_no_nan(self):
        x = make_audio_batch(BATCH)
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"


class TestTSMixerEncoder:
    def setup_method(self):
        from src.encoders.tsmixer.model import TSMixerEncoder
        self.model = TSMixerEncoder(seq_len=HUMIDITY_SEQ_LEN, in_channels=1, embed_dim=EMBED_DIM)
        self.model.eval()

    def test_output_shape(self):
        x = make_humidity_batch(BATCH)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (BATCH, EMBED_DIM), f"Expected ({BATCH}, {EMBED_DIM}), got {out.shape}"

    def test_no_nan(self):
        x = make_humidity_batch(BATCH)
        with torch.no_grad():
            out = self.model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"


# ---------------------------------------------------------------------------
# Fusion smoke tests (skipped until encoders are implemented)
# ---------------------------------------------------------------------------

IMU_DIM = EMBED_DIM
AUDIO_DIM = 768
HUM_DIM = EMBED_DIM


@pytest.fixture
def fake_embeddings():
    return (
        torch.randn(BATCH, IMU_DIM),
        torch.randn(BATCH, AUDIO_DIM),
        torch.randn(BATCH, HUM_DIM),
    )


class TestLateFusion:
    def test_output_shape(self, fake_embeddings):
        from src.fusion.late_fusion import LateFusion
        model = LateFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)
        imu, audio, hum = fake_embeddings
        with torch.no_grad():
            out = model(imu, audio, hum)
        assert out.shape == (BATCH, NUM_CLASSES)


class TestGMF:
    def test_output_shape(self, fake_embeddings):
        from src.fusion.gmf import GatedMultimodalFusion
        model = GatedMultimodalFusion(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)
        imu, audio, hum = fake_embeddings
        with torch.no_grad():
            out = model(imu, audio, hum)
        assert out.shape == (BATCH, NUM_CLASSES)


class TestMBT:
    def test_output_shape(self, fake_embeddings):
        from src.fusion.mbt import MultimodalBottleneckTransformer
        model = MultimodalBottleneckTransformer(IMU_DIM, AUDIO_DIM, HUM_DIM, NUM_CLASSES)
        imu, audio, hum = fake_embeddings
        with torch.no_grad():
            out = model(imu, audio, hum)
        assert out.shape == (BATCH, NUM_CLASSES)
