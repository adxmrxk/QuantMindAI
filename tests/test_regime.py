from itertools import permutations

import numpy as np

from quantmind.data import generate_regime_series
from quantmind.models import analyze


def _best_accuracy(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Accuracy under the best mapping of predicted states -> true regimes."""
    best = 0.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in pred])
        best = max(best, float((mapped == true).mean()))
    return best


def test_analyze_output_contract():
    close = generate_regime_series(n_days=800, seed=7)["close"]
    result = analyze(close, n_states=3)

    assert set(result.frame.columns) == {"close", "ret", "mom", "vol", "state", "label", "confidence"}
    assert set(result.label_legend.values()) == {"Bear", "Neutral", "Bull"}
    # Confidence is a valid probability.
    assert result.frame["confidence"].between(0.0, 1.0).all()
    # current() reports the last row.
    assert result.current["label"] in {"Bear", "Neutral", "Bull"}


def test_regimes_are_persistent_not_flickering():
    """An HMM should produce sticky regimes, not relabel every other day."""
    close = generate_regime_series(n_days=1000, seed=7)["close"]
    states = analyze(close, n_states=3).frame["state"].to_numpy()
    switches = int((states[1:] != states[:-1]).sum())
    # Far fewer switches than days => persistent regimes.
    assert switches < len(states) * 0.2


def test_recovers_known_synthetic_regimes():
    df = generate_regime_series(n_days=1500, seed=7)
    result = analyze(df["close"], n_states=3)

    # Align the ground truth to whatever index the pipeline produced.
    true = df["true_regime"].loc[result.frame.index].to_numpy()
    pred = result.frame["state"].to_numpy()

    acc = _best_accuracy(true, pred, k=3)
    # Well-separated synthetic regimes should be recovered comfortably.
    assert acc > 0.7, f"regime recovery accuracy too low: {acc:.2f}"


def test_label_ordering_matches_returns():
    """Bull label must sit on the highest-mean-return state, Bear on the lowest."""
    close = generate_regime_series(n_days=900, seed=7)["close"]
    result = analyze(close, n_states=3)
    frame = result.frame

    avg_by_label = frame.groupby("label")["ret"].mean()
    assert avg_by_label["Bull"] > avg_by_label["Neutral"] > avg_by_label["Bear"]
