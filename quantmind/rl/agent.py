"""Train and roll out a PPO agent over the portfolio environment."""

from __future__ import annotations

import pandas as pd

from quantmind.features import compute_features
from quantmind.rl.env import PortfolioEnv


def train_ppo(close: pd.Series, timesteps: int = 5000, seed: int = 0, **env_kwargs):
    """Train a PPO policy on the portfolio environment. Returns the SB3 model."""
    from stable_baselines3 import PPO

    env = PortfolioEnv(close, **env_kwargs)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        n_steps=256,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
    )
    model.learn(total_timesteps=timesteps)
    return model


def policy_weights(model, close: pd.Series, vol_window: int = 20, **env_kwargs) -> pd.Series:
    """Roll the trained policy across the series and return its daily weights.

    The resulting Series can be fed straight into ``run_backtest`` as positions.
    """
    env = PortfolioEnv(close, vol_window=vol_window, **env_kwargs)
    index = compute_features(close, vol_window=vol_window).index

    obs, _ = env.reset()
    weights = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        weights.append(info["weight"])
        if terminated or truncated:
            break
    return pd.Series(weights, index=index[: len(weights)], name="position")
