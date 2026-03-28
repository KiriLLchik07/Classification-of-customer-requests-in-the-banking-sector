import pytest
from pydantic import ValidationError

from services.backend.app.schemas.request import PredictRequest

def test_predict_request_uses_default_model_settings():
    request = PredictRequest(text="check card delivery")

    assert request.model_name == "Banking77_Classifier"
    assert request.model_stage == "Production"

def test_predict_request_strips_whitespace():
    request = PredictRequest(text="card not working")

    assert request.text == "card not working"

def test_predict_request_rejects_whitespace_only_text():
    with pytest.raises(ValidationError):
        PredictRequest(text=" ")
