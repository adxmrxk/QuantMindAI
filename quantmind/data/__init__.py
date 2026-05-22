"""Data layer: market price ingestion and synthetic data generation."""

from quantmind.data.loader import load_prices
from quantmind.data.synthetic import generate_regime_series

__all__ = ["load_prices", "generate_regime_series"]
