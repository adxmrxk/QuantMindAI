import pandas as pd

from quantmind.data import generate_regime_series
from quantmind.features import compute_features


def test_synthetic_series_shape_and_determinism():
    a = generate_regime_series(n_days=500, seed=7)
    b = generate_regime_series(n_days=500, seed=7)

    assert list(a.columns) == ["close", "true_regime"]
    assert len(a) == 500
    assert isinstance(a.index, pd.DatetimeIndex)
    assert (a["close"] > 0).all()
    # Same seed -> identical series.
    pd.testing.assert_frame_equal(a, b)
    # All three regimes should appear over a long-enough window.
    assert set(a["true_regime"].unique()) == {0, 1, 2}


def test_compute_features_drops_warmup_and_scales():
    close = generate_regime_series(n_days=300, seed=1)["close"]
    feats = compute_features(close, vol_window=10)

    assert list(feats.columns) == ["ret", "mom", "vol"]
    # Lose 1 row to the return diff plus (vol_window - 1) to the rolling stats.
    assert len(feats) == len(close) - 10
    assert not feats.isna().any().any()
    assert (feats["vol"] >= 0).all()
