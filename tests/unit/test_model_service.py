from unittest.mock import Mock

import pandas as pd
import pytest

from services.backend.app.services.model_service import ModelService

def test_load_model_uses_mlflow_registry_uri(monkeypatch):
    service = ModelService()
    fake_model = Mock()
    captured = {}

    def fake_load_model(model_uri: str):
        captured["model_uri"] = model_uri
        return fake_model

    monkeypatch.setattr("app.services.model_service.mlflow.pyfunc.load_model", fake_load_model)

    result = service.load_model("Banking77_Classifier", stage="Staging")

    assert result is fake_model
    assert captured["model_uri"] == "models:/Banking77_Classifier/Staging"
    assert service.get_model("Banking77_Classifier") is fake_model

def test_load_model_reraises_mlflow_errors(monkeypatch):
    service = ModelService()

    def fake_load_model(model_uri: str):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("app.services.model_service.mlflow.pyfunc.load_model", fake_load_model)

    with pytest.raises(RuntimeError, match="registry unavailable"):
        service.load_model("Banking77_Classifier")

def test_predict_uses_dataframe_and_returns_first_prediction():
    service = ModelService()
    fake_model = Mock()
    fake_model.predict.return_value = pd.DataFrame({"prediction": ["cash_withdrawal"]})
    service.models["Banking77_Classifier"] = fake_model

    prediction = service.predict("Need cash", "Banking77_Classifier")

    assert prediction == "cash_withdrawal"
    fake_model.predict.assert_called_once()
    model_input = fake_model.predict.call_args.args[0]
    assert list(model_input.columns) == ["text"]
    assert model_input.iloc[0, 0] == "Need cash"

def test_get_model_raises_for_unknown_model():
    service = ModelService()

    with pytest.raises(ValueError, match="not loaded"):
        service.get_model("missing-model")
