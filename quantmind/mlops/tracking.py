"""Experiment tracking with an optional MLflow backend."""

from __future__ import annotations

import contextlib
import os


def mlflow_enabled() -> bool:
    return os.getenv("QUANTMIND_MLFLOW", "0") == "1"


class _Run:
    """Uniform interface for both backends."""

    def __init__(self, mlflow=None) -> None:
        self._mlflow = mlflow
        self.params: dict = {}
        self.metrics: dict = {}

    def log_params(self, params: dict) -> None:
        self.params.update(params)
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(metrics)
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics)


@contextlib.contextmanager
def track_run(name: str, params: dict | None = None):
    """Yield a run object exposing ``log_params`` / ``log_metrics``.

    Uses MLflow if enabled and importable; falls back to an in-memory no-op run
    otherwise. Safe to call in tests and CI without MLflow installed.
    """
    params = params or {}
    if mlflow_enabled():
        try:
            import mlflow

            mlflow.set_experiment(os.getenv("QUANTMIND_EXPERIMENT", "quantmind"))
            with mlflow.start_run(run_name=name):
                run = _Run(mlflow)
                run.log_params(params)
                yield run
                return
        except ImportError:
            pass  # fall through to no-op

    run = _Run(None)
    run.log_params(params)
    yield run
