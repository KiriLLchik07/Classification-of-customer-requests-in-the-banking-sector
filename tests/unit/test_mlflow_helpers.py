from pathlib import Path

from src.mlops.mlflow.registry import register_model
from src.mlops.mlflow.tracking import log_experiment
from src.mlops.packaging.log_pyfunc_model import log_pyfunc_model

def test_log_experiment_logs_metrics_params_and_optional_model(monkeypatch):
    events = []

    monkeypatch.setattr("src.mlops.mlflow.tracking.mlflow.log_params", lambda params: events.append(("params", params)))
    monkeypatch.setattr("src.mlops.mlflow.tracking.mlflow.log_metrics", lambda metrics: events.append(("metrics", metrics)))
    monkeypatch.setattr(
        "src.mlops.mlflow.tracking.mlflow.pytorch.log_model",
        lambda model, artifact_path: events.append(("model", model, artifact_path)),
    )
    monkeypatch.setattr(
        "src.mlops.mlflow.tracking.mlflow.log_artifact",
        lambda path, artifact_path=None: events.append(("artifact", path, artifact_path)),
    )

    log_experiment(
        model="model-object",
        metrics={"f1_macro": 0.91},
        params={"epochs": 3},
        artifacts={"report": "artifacts/reports/report.csv"},
        model_artifact_path="pytorch-model",
    )

    assert ("params", {"epochs": 3}) in events
    assert ("metrics", {"f1_macro": 0.91}) in events
    assert ("model", "model-object", "pytorch-model") in events
    assert ("artifact", "artifacts/reports/report.csv", "report") in events

def test_register_model_returns_registered_version(monkeypatch):
    class DummyRegisteredModel:
        version = 17

    monkeypatch.setattr(
        "src.mlops.mlflow.registry.mlflow.register_model",
        lambda model_uri, name: DummyRegisteredModel(),
    )

    version = register_model("run-123", "model", "Banking77_Classifier")

    assert version == 17

def test_log_pyfunc_model_uses_requested_artifact_path(monkeypatch):
    captured = {}

    def fake_log_model(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.mlops.packaging.log_pyfunc_model.mlflow.pyfunc.log_model", fake_log_model)

    log_pyfunc_model(str(Path("artifacts/local_models/transformers")), artifact_path="serving-model")

    assert captured["artifact_path"] == "serving-model"
    assert Path(captured["artifacts"]["model"]).as_posix() == "artifacts/local_models/transformers"
