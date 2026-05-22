"""Forward-looking regime forecasting with a time-series Transformer.

The HMM in :mod:`quantmind.models` is a *smoother* — it labels the regime each
day given the whole series. This module is a *predictor*: a Transformer encoder
that, from a trailing window of (return, momentum, volatility) features,
forecasts the regime for the next day. Together they mirror how desks operate —
a backward-looking state estimate plus a forward-looking forecast.
"""

from quantmind.forecast.train import ForecastModel, train_forecaster
from quantmind.forecast.transformer import RegimeTransformer

__all__ = ["ForecastModel", "train_forecaster", "RegimeTransformer"]
