from pydantic import BaseModel, Field
from typing import Optional

class PredictResponse(BaseModel):
    """Схема ответа для эндпоинта /predict"""
    prediction: str = Field(
        ...,
        description="Предсказанный класс запроса"
    )
    model_name: str = Field(
        ...,
        description="Название модели, использованной для предсказания"
    )
    confidence: Optional[float] = Field(
        None,
        description="Уверенность модели в диапазоне (0-1)",
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
