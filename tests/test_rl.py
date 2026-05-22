import numpy as np

from quantmind.backtest import run_backtest
from quantmind.data import generate_regime_series
from quantmind.rl import PortfolioEnv, policy_weights, train_ppo


def _close():
    return generate_regime_series(n_days=600, seed=5)["close"]


def test_env_conforms_to_gymnasium_api():
    from gymnasium.utils.env_checker import check_env

    # Raises if the env violates the Gymnasium contract.
    check_env(PortfolioEnv(_close()), skip_render_check=True)


def test_env_step_contract():
    env = PortfolioEnv(_close())
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)

    obs, reward, terminated, truncated, info = env.step([0.7])
    assert obs.shape == (4,)
    assert np.isfinite(reward)
    assert 0.0 <= info["weight"] <= 1.0
    assert isinstance(terminated, bool)


def test_env_runs_to_termination():
    env = PortfolioEnv(_close())
    env.reset()
    steps, terminated = 0, False
    while not terminated and steps < 10_000:
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert terminated
    assert steps == env.n - 1


def test_ppo_trains_and_is_backtestable():
    close = _close()
    model = train_ppo(close, timesteps=1200, seed=0)
    weights = policy_weights(model, close)

    assert weights.between(0.0, 1.0).all()
    bt = run_backtest(close.loc[weights.index], weights, cost_bps=1.0)
    assert all(np.isfinite(v) for v in bt.metrics.values())
