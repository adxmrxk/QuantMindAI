"""Lightweight MLOps helpers: experiment tracking that degrades gracefully.

``track_run`` is a context manager that logs params/metrics to MLflow when it is
installed and ``QUANTMIND_MLFLOW=1``; otherwise it is a no-op with the same
interface, so training code can call it unconditionally.
"""

from quantmind.mlops.tracking import mlflow_enabled, track_run

__all__ = ["track_run", "mlflow_enabled"]
