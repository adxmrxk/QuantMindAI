import numpy as np
import torch

from quantmind.data import generate_regime_series
from quantmind.forecast import RegimeTransformer, train_forecaster
from quantmind.forecast.dataset import build_dataset


def test_transformer_forward_shape():
    net = RegimeTransformer(n_features=3, seq_len=30, n_classes=3)
    x = torch.randn(8, 30, 3)
    out = net(x)
    assert out.shape == (8, 3)


def test_build_dataset_shapes():
    close = generate_regime_series(n_days=600, seed=2)["close"]
    X, y, states = build_dataset(close, n_states=3, seq_len=30)
    assert X.ndim == 3 and X.shape[1] == 30 and X.shape[2] == 3
    assert len(X) == len(y)
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_forecaster_trains_and_beats_majority():
    close = generate_regime_series(n_days=2000, seed=7)["close"]
    model, metrics = train_forecaster(close, n_states=3, seq_len=30, epochs=12, seed=0)

    # Must learn something beyond guessing the most common class.
    assert metrics["val_accuracy"] >= metrics["majority_baseline"]
    assert metrics["val_accuracy"] > 0.6

    # Inference contract: a single window -> a valid probability vector.
    X, _, _ = build_dataset(close, n_states=3, seq_len=30)
    proba = model.predict_proba(X[-1])
    assert proba.shape == (1, 3)
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)
