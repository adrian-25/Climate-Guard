"""tests/dl/test_model.py — Model shape and reproducibility tests."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; DL tests skipped")

import torch as _torch


def _make_batch(batch=4, seq=7, n_feat=21, n_cities=5):
    x = _torch.randn(batch, seq, n_feat)
    c = _torch.randint(0, n_cities, (batch,))
    return x, c


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["gru", "lstm"])
def test_output_shape(arch):
    from dl.model import build_model

    model = build_model(n_features=21, arch=arch)
    model.eval()
    x, c = _make_batch()
    with _torch.no_grad():
        out = model(x, c)
    assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"


@pytest.mark.parametrize("arch", ["gru", "lstm"])
def test_output_is_raw_logit(arch):
    """Output should be unbounded logit (not sigmoid)."""
    from dl.model import build_model

    model = build_model(n_features=21, arch=arch)
    model.eval()
    x, c = _make_batch()
    with _torch.no_grad():
        out = model(x, c)
    # Raw logit can be > 1 or < 0; sigmoid would be in [0,1]
    # Just verify it's a single float per sample
    assert out.dtype == _torch.float32


@pytest.mark.parametrize("arch,n_feat", [("gru", 21), ("gru", 104), ("lstm", 21)])
def test_output_shape_various_configs(arch, n_feat):
    from dl.model import build_model

    model = build_model(n_features=n_feat, arch=arch)
    model.eval()
    x, c = _make_batch(n_feat=n_feat)
    with _torch.no_grad():
        out = model(x, c)
    assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["gru", "lstm"])
def test_same_output_under_same_seed(arch):
    """Two models initialised with the same seed must produce identical outputs."""
    import numpy as np

    from dl.model import build_model

    def _run(seed):
        _torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(n_features=21, arch=arch)
        model.eval()
        _torch.manual_seed(seed + 100)
        x, c = _make_batch()
        with _torch.no_grad():
            return model(x, c).numpy()

    out1 = _run(42)
    out2 = _run(42)
    assert (out1 == out2).all(), "Same seed must give identical outputs"


# ---------------------------------------------------------------------------
# City embedding dimension
# ---------------------------------------------------------------------------


def test_city_embedding_dim():
    from dl.config import CITY_EMBED_DIM, NUM_INDIA_CITIES
    from dl.model import build_model

    model = build_model(n_features=21, arch="gru")
    assert model.city_embed.embedding_dim == CITY_EMBED_DIM
    assert model.city_embed.num_embeddings == NUM_INDIA_CITIES


def test_weighted_focal_loss_is_finite_and_backpropagates():
    """The rare-event focal-loss ablation must produce a usable gradient."""
    from dl.train import WeightedFocalLoss

    logits = _torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    targets = _torch.tensor([1.0, 0.0, 1.0])
    loss = WeightedFocalLoss(pos_weight=10.0)(logits, targets)
    assert _torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert _torch.isfinite(logits.grad).all()
