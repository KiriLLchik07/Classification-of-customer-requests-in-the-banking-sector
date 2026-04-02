from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_service import model_service

@pytest.fixture
def client(monkeypatch):
    original_models = dict(model_service.models)
    model_service.models.clear()

    def fake_startup_load(model_name: str, stage: str = "Production"):
        model_service.models[model_name] = Mock(name=f"{model_name}:{stage}")
        return model_service.models[model_name]

    monkeypatch.setattr("app.main.model_service.load_model", fake_startup_load)

    with TestClient(app) as test_client:
        yield test_client

    model_service.models.clear()
    model_service.models.update(original_models)

def test_health_returns_ok_when_model_loaded_on_startup(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == ["Banking77_Classifier"]

def test_models_endpoint_returns_loaded_models(client):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["Banking77_Classifier"]}

def test_predict_uses_loaded_model_and_returns_prediction(client, monkeypatch):
    monkeypatch.setattr(model_service, "predict", lambda text, model_name: "cash_withdrawal")

    response = client.post(
        "/api/v1/predict",
        json={"text": "Where is my cash withdrawal?", "model_name": "Banking77_Classifier"},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "cash_withdrawal"

def test_predict_loads_missing_model_with_requested_stage(client, monkeypatch):
    model_service.models.clear()
    captured = {}

    def fake_load_model(model_name: str, stage: str = "Production"):
        captured["model_name"] = model_name
        captured["stage"] = stage
        model_service.models[model_name] = Mock(name=f"{model_name}:{stage}")
        return model_service.models[model_name]

    monkeypatch.setattr(model_service, "load_model", fake_load_model)
    monkeypatch.setattr(model_service, "predict", lambda text, model_name: "balance_not_updated")

    response = client.post(
        "/api/v1/predict",
        json={
            "text": "My balance is not updated",
            "model_name": "CandidateModel",
            "model_stage": "Staging",
        },
    )

    assert response.status_code == 200
    assert captured == {"model_name": "CandidateModel", "stage": "Staging"}

def test_predict_returns_400_when_model_load_fails(client, monkeypatch):
    model_service.models.clear()
    monkeypatch.setattr(model_service, "load_model", lambda model_name, stage="Production": (_ for _ in ()).throw(RuntimeError("registry unavailable")))

    response = client.post(
        "/api/v1/predict",
        json={
            "text": "My card payment failed",
            "model_name": "BrokenModel",
            "model_stage": "Production",
        },
    )

    assert response.status_code == 400
    assert "not available" in response.json()["detail"]

def test_predict_returns_500_when_prediction_fails(client, monkeypatch):
    monkeypatch.setattr(model_service, "predict", lambda text, model_name: (_ for _ in ()).throw(RuntimeError("inference crashed")))

    response = client.post(
        "/api/v1/predict",
        json={"text": "Transfer not received", "model_name": "Banking77_Classifier"},
    )

    assert response.status_code == 500
    assert "Prediction_failed" in response.json()["detail"]

def test_health_returns_503_when_no_models_loaded(client):
    model_service.models.clear()

    response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "unhealthy"

def test_model_info_returns_404_when_registry_has_no_versions(client, monkeypatch):
    class DummyClient:
        def get_latest_versions(self, model_name: str):
            return []

    monkeypatch.setattr("app.api.v1.model_info.mlflow.tracking.MlflowClient", lambda tracking_uri=None: DummyClient())

    response = client.get("/api/v1/model_info/UnknownModel")

    assert response.status_code == 404

def test_model_info_returns_versions(client, monkeypatch):
    version = Mock(version="12", current_stage="Production", description="best model")

    class DummyClient:
        def get_latest_versions(self, model_name: str):
            return [version]

    monkeypatch.setattr("app.api.v1.model_info.mlflow.tracking.MlflowClient", lambda tracking_uri=None: DummyClient())

    response = client.get("/api/v1/model_info/Banking77_Classifier")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "Banking77_Classifier",
        "versions": [
            {
                "version": "12",
                "stage": "Production",
                "description": "best model",
            }
        ],
    }

def test_model_info_returns_500_when_registry_request_fails(client, monkeypatch):
    class DummyClient:
        def get_latest_versions(self, model_name: str):
            raise RuntimeError("mlflow unavailable")

    monkeypatch.setattr("app.api.v1.model_info.mlflow.tracking.MlflowClient", lambda tracking_uri=None: DummyClient())

    response = client.get("/api/v1/model_info/Banking77_Classifier")

    assert response.status_code == 500
    assert response.json()["detail"] == "mlflow unavailable"
