"""A single-asset portfolio allocation environment (Gymnasium API).

State    : z-scored (return, momentum, volatility) for the current day plus the
           agent's current asset weight.
Action   : a continuous target weight in [0, 1] (fraction in the risky asset;
           the rest is cash).
Reward   : next-day portfolio return (scaled to %), minus a transaction-cost
           charge on turnover and a risk penalty proportional to weight × local
           volatility. This rewards capturing upside while discouraging holding
           risk into turbulent (high-vol) regimes.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from quantmind.features import compute_features

FEATURES = ["ret", "mom", "vol"]


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        close: pd.Series,
        risk_aversion: float = 0.05,
        cost_bps: float = 1.0,
        vol_window: int = 20,
    ) -> None:
        super().__init__()
        feats = compute_features(close, vol_window=vol_window)
        prices = pd.Series(close, dtype="float64").loc[feats.index].to_numpy()

        self._prices = prices
        self._returns = np.diff(prices) / prices[:-1]  # length n-1
        self._feat = feats[FEATURES].to_numpy(dtype="float32")
        self._fmean = self._feat.mean(axis=0)
        self._fstd = self._feat.std(axis=0) + 1e-8

        self.risk_aversion = risk_aversion
        self.cost = cost_bps / 1e4
        self.n = len(prices)

        # Three z-scored features (clipped to ±10) plus the current weight in [0, 1].
        low = np.array([-10.0, -10.0, -10.0, 0.0], dtype=np.float32)
        high = np.array([10.0, 10.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._t = 0
        self._w = 0.0

    def _obs(self) -> np.ndarray:
        f = np.clip((self._feat[self._t] - self._fmean) / self._fstd, -10.0, 10.0)
        return np.array([f[0], f[1], f[2], self._w], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._w = 0.0
        return self._obs(), {}

    def step(self, action):
        w = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        r = float(self._returns[self._t]) if self._t < len(self._returns) else 0.0
        turnover = abs(w - self._w)
        port_ret = w * r - self.cost * turnover
        local_vol = float(self._feat[self._t, 2])

        reward = port_ret * 100.0 - self.risk_aversion * w * abs(local_vol)

        self._w = w
        self._t += 1
        terminated = self._t >= self.n - 1
        return self._obs(), float(reward), terminated, False, {"weight": w, "port_ret": port_ret}
