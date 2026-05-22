"""Reinforcement-learning portfolio allocation.

A Gymnasium environment in which an agent chooses, each day, what fraction of
capital to hold in a risky asset versus cash, rewarded on risk-adjusted return
net of transaction costs. Trained with PPO (Stable-Baselines3).
"""

from quantmind.rl.agent import policy_weights, train_ppo
from quantmind.rl.env import PortfolioEnv

__all__ = ["PortfolioEnv", "train_ppo", "policy_weights"]
