"""Market regime detection with a Gaussian Hidden Markov Model.

Why an HMM? A regime is a *persistent latent state* that emits observable
(return, volatility) pairs. An HMM models exactly that: hidden states with
Gaussian emissions and sticky transitions, so the labels it produces are
temporally smooth rather than flickering day to day — which is what makes them
useful as "regimes" instead of per-day classifications.

The raw HMM states are anonymous integers; we map them to interpretable labels
(Bear / Neutral / Bull) by ordering states on their mean return. The state with
the lowest mean return is Bear, the highest is Bull, the rest Neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.features.indicators import FEATURE_COLUMNS, compute_features

MODEL_STORE = Path(__file__).resolve().parents[2] / "models" / "store"


class RegimeModel:
    """Thin wrapper around hmmlearn's GaussianHMM with regime labelling."""

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_init = n_init
        self._hmm = None  # lazily constructed in fit()
        self._score = None  # log-likelihood of the chosen fit

    def fit(self, features: np.ndarray) -> "RegimeModel":
        """Fit with several random restarts, keeping the highest-likelihood run.

        EM for HMMs converges to local optima and readily falls into degenerate
        solutions (e.g. two identical states the chain flips between). Restarting
        from different seeds and selecting on log-likelihood avoids that.
        """
        from hmmlearn.hmm import GaussianHMM

        X = np.asarray(features, dtype="float64")
        best, best_score = None, -np.inf
        for i in range(self.n_init):
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state + i,
            )
            try:
                model.fit(X)
                score = model.score(X)
            except Exception:  # noqa: BLE001 - a bad restart shouldn't kill the fit
                continue
            if np.isfinite(score) and score > best_score:
                best, best_score = model, score

        if best is None:
            raise RuntimeError("HMM failed to converge on every restart.")
        self._hmm = best
        self._score = float(best_score)
        return self

    def _check_fitted(self) -> None:
        if self._hmm is None:
            raise RuntimeError("RegimeModel is not fitted yet — call fit() first.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Most-likely state sequence (Viterbi)."""
        self._check_fitted()
        return self._hmm.predict(np.asarray(features, dtype="float64"))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Posterior state probabilities, shape (n_samples, n_states)."""
        self._check_fitted()
        return self._hmm.predict_proba(np.asarray(features, dtype="float64"))

    def state_labels(self) -> dict[int, str]:
        """Map each anonymous HMM state to a human regime label by mean return."""
        self._check_fitted()
        mean_return = self._hmm.means_[:, 0]  # column 0 == "ret"
        order = np.argsort(mean_return)  # ascending: most bearish first
        labels: dict[int, str] = {}
        for rank, state in enumerate(order):
            if rank == 0:
                labels[int(state)] = "Bear"
            elif rank == len(order) - 1:
                labels[int(state)] = "Bull"
            else:
                labels[int(state)] = "Neutral"
        return labels

    def save(self, name: str) -> Path:
        import joblib

        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        path = MODEL_STORE / f"{name}.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(name: str) -> "RegimeModel":
        import joblib

        return joblib.load(MODEL_STORE / f"{name}.joblib")


@dataclass
class RegimeResult:
    """Output of :func:`analyze` — everything the API/dashboard needs."""

    frame: pd.DataFrame  # close, state, label, confidence (indexed by date)
    label_legend: dict[int, str]  # state index -> label
    model: RegimeModel

    @property
    def current(self) -> dict:
        last = self.frame.iloc[-1]
        return {
            "date": self.frame.index[-1].strftime("%Y-%m-%d"),
            "state": int(last["state"]),
            "label": str(last["label"]),
            "confidence": float(last["confidence"]),
        }

    def regime_stats(self) -> dict[str, dict]:
        """Per-regime summary: day count, average return and volatility."""
        stats: dict[str, dict] = {}
        for label, grp in self.frame.groupby("label"):
            stats[str(label)] = {
                "days": int(len(grp)),
                "share": round(len(grp) / len(self.frame), 3),
                "avg_return": round(float(grp["ret"].mean()), 4),
                "avg_vol": round(float(grp["vol"].mean()), 4),
            }
        return stats


def analyze(
    close: pd.Series | pd.DataFrame,
    n_states: int = 3,
    vol_window: int = 20,
    random_state: int = 42,
) -> RegimeResult:
    """Run the full regime pipeline on a close-price series.

    data -> features -> fit HMM -> predict states -> label them.
    """
    if isinstance(close, pd.DataFrame):
        close = close["close"]
    close = pd.Series(close, dtype="float64")

    feats = compute_features(close, vol_window=vol_window)
    X = feats[FEATURE_COLUMNS].to_numpy()

    model = RegimeModel(n_states=n_states, random_state=random_state).fit(X)
    states = model.predict(X)
    proba = model.predict_proba(X)
    legend = model.state_labels()

    frame = pd.DataFrame(index=feats.index)
    frame["close"] = close.loc[feats.index]
    frame["ret"] = feats["ret"]
    frame["mom"] = feats["mom"]
    frame["vol"] = feats["vol"]
    frame["state"] = states
    frame["label"] = [legend[int(s)] for s in states]
    frame["confidence"] = proba.max(axis=1)

    return RegimeResult(frame=frame, label_legend=legend, model=model)
