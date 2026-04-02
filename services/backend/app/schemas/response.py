from pydantic import BaseModel, Field
from typing import Optional

class PredictResponse(BaseModel):
    """Схема ответа для эндпоинта /predict"""
    prediction: str = Field(
        ...,
        description="Predicted query class"
    )
    model_name: str = Field(
        ...,
        description="The name of the model used for prediction"
    )
    confidence: Optional[float] = Field(
        None,
        description="The confidence of the model in the range (0-1)",
        ge=0.0,
        le=1.0,
    )

class HealthResponce(BaseModel):
    """Схема ответа для эндпоинта /health"""
    status: str = Field(...)
    service: str = Field(...)
    version: str = Field(...)
    models_loaded: list[str] = Field(default_factory=list)

class ModelInfoResponce(BaseModel):
    """Схема ответа для эндпоинта /model_info"""
    model_name: str
    version: str
    stage: str
    description: Optional[str] = None
