from quantmind.mlops import mlflow_enabled, track_run


def test_tracking_is_noop_without_mlflow(monkeypatch):
    monkeypatch.setenv("QUANTMIND_MLFLOW", "0")
    assert mlflow_enabled() is False

    with track_run("unit-test", {"lr": 0.01}) as run:
        run.log_metrics({"accuracy": 0.9})

    # The no-op run still records what it was given, in memory.
    assert run.params["lr"] == 0.01
    assert run.metrics["accuracy"] == 0.9


def test_mlflow_enabled_flag(monkeypatch):
    monkeypatch.setenv("QUANTMIND_MLFLOW", "1")
    assert mlflow_enabled() is True
    monkeypatch.setenv("QUANTMIND_MLFLOW", "0")
    assert mlflow_enabled() is False
