from unittest.mock import Mock

import pandas as pd
import pytest
from config.settings import settings

from services.backend.app.services.model_service import ModelService

def test_load_model_uses_mlflow_registry_uri(monkeypatch):
    service = ModelService()
    fake_model = Mock()
    captured = {}

    def fake_load_model(model_uri: str):
        captured["model_uri"] = model_uri
        return fake_model

    monkeypatch.setattr("services.backend.app.services.model_service.mlflow.pyfunc.load_model", fake_load_model)

    result = service.load_model("Banking77_LogisticRegression", alias="baseline")

    assert result is fake_model
    assert captured["model_uri"] == "models:/Banking77_LogisticRegression@baseline"
    assert service.get_model("Banking77_LogisticRegression") is fake_model
    assert service.get_loaded_alias("Banking77_LogisticRegression") == "baseline"

def test_load_model_reraises_mlflow_errors(monkeypatch):
    service = ModelService()

    def fake_load_model(model_uri: str):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("services.backend.app.services.model_service.mlflow.pyfunc.load_model", fake_load_model)

    with pytest.raises(RuntimeError, match="registry unavailable"):
        service.load_model(settings.model_name)

def test_predict_uses_dataframe_and_returns_first_prediction():
    service = ModelService()
    fake_model = Mock()
    fake_model.predict.return_value = pd.DataFrame({"prediction": ["cash_withdrawal"]})
    service.models[settings.model_name] = fake_model

    prediction = service.predict("Need cash", settings.model_name)

    assert prediction == "cash_withdrawal"
    fake_model.predict.assert_called_once()
    model_input = fake_model.predict.call_args.args[0]
    assert list(model_input.columns) == ["text"]
    assert model_input.iloc[0, 0] == "Need cash"

def test_get_model_raises_for_unknown_model():
    service = ModelService()

    with pytest.raises(ValueError, match="not loaded"):
        service.get_model("missing-model")

def test_resolve_prediction_maps_numeric_code_to_label():
    service = ModelService()
    service.code_to_label = {45: "card_arrival"}
    service.label_to_code = {"card_arrival": 45}

    label, code = service.resolve_prediction("45")

    assert label == "card_arrival"
    assert code == 45

def test_resolve_prediction_maps_label_to_code_case_insensitive():
    service = ModelService()
    service.code_to_label = {45: "card_arrival"}
    service.label_to_code = {"card_arrival": 45}

    label, code = service.resolve_prediction("Card_Arrival")

    assert label == "card_arrival"
    assert code == 45
