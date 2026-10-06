"""tests/dl/test_evaluate.py — Metric and bootstrap correctness tests."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; DL tests skipped")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric range tests on small fixture
# ---------------------------------------------------------------------------


def _fixture():
    rng = np.random.default_rng(0)
    n = 200
    labels = (rng.random(n) > 0.9).astype(float)  # ~10% positive
    probs = rng.random(n).astype(np.float32)
    meta = pd.DataFrame(
        {
            "city_key": ["delhi"] * 100 + ["lucknow"] * 100,
            "date": [f"2023-{1+(i//30):02d}-{1+(i%30):02d}" for i in range(n)],
        }
    )
    return labels, probs, meta


def test_metrics_in_valid_range():
    from dl.evaluate import compute_metrics

    labels, probs, _ = _fixture()
    m = compute_metrics(labels, probs, threshold=0.5)
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0


def test_metrics_na_for_all_negative():
    """pr_auc and roc_auc should be None when there are no positives."""
    from dl.evaluate import compute_metrics

    labels = np.zeros(50, dtype=float)
    probs = np.random.default_rng(1).random(50).astype(np.float32)
    m = compute_metrics(labels, probs, threshold=0.5)
    assert m["pr_auc"] is None
    assert m["roc_auc"] is None


def test_per_city_na_for_no_positives():
    """Cities with no positives must return None (n/a), not a metrics dict."""
    from dl.evaluate import per_city_metrics

    labels, probs, meta = _fixture()
    # Force lucknow to have no positives
    lucknow_mask = meta["city_key"] == "lucknow"
    labels[lucknow_mask.values] = 0.0
    result = per_city_metrics(labels, probs, meta, threshold=0.5)
    assert result.get("lucknow") is None, "lucknow should be n/a with no positives"
    assert result.get("delhi") is not None, "delhi has positives, should have metrics"


# ---------------------------------------------------------------------------
# Bootstrap CI correctness
# ---------------------------------------------------------------------------


def test_bootstrap_returns_ordered_interval():
    """Lower bound must be <= upper bound."""
    from dl.evaluate import block_bootstrap_ci

    labels, probs, meta = _fixture()
    ci = block_bootstrap_ci(
        labels_ref=labels,
        probs_ref=probs,
        labels_cmp=labels,
        probs_cmp=probs,
        meta=meta,
        block_size=10,
        n_resamples=100,
        threshold_ref=0.5,
        threshold_cmp=0.5,
    )
    if ci["f1_ci_lower"] is not None:
        assert ci["f1_ci_lower"] <= ci["f1_ci_upper"]
    if ci["prauc_ci_lower"] is not None:
        assert ci["prauc_ci_lower"] <= ci["prauc_ci_upper"]


def test_bootstrap_identical_models_interval_includes_zero():
    """Identical models should produce a CI centred near 0."""
    from dl.evaluate import block_bootstrap_ci

    labels, probs, meta = _fixture()
    ci = block_bootstrap_ci(
        labels_ref=labels,
        probs_ref=probs,
        labels_cmp=labels,
        probs_cmp=probs,
        meta=meta,
        block_size=10,
        n_resamples=200,
        threshold_ref=0.5,
        threshold_cmp=0.5,
    )
    assert ci["includes_zero_f1"], "Identical models: F1 CI must include 0"
    assert ci["includes_zero_prauc"], "Identical models: PR-AUC CI must include 0"
