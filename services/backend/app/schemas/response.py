from typing import Optional
from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted query class")
    prediction_label: str = Field(..., description="Human-readable predicted category")
    prediction_code: Optional[int] = Field(None, description="Predicted category code")
    model_name: str = Field(..., description="Model used for prediction")
    confidence: Optional[float] = Field(
        None,
        description="Model confidence in range [0, 1]",
        ge=0.0,
        le=1.0,
    )

class HealthResponce(BaseModel):
    status: str = Field(...)
    service: str = Field(...)
    version: str = Field(...)
    models_loaded: list[str] = Field(default_factory=list)

class ModelInfoVersion(BaseModel):
    version: str
    alias: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_name: str
    versions: list[ModelInfoVersion] = Field(default_factory=list)

class MlflowModelsResponse(BaseModel):
    model_names: list[str] = Field(default_factory=list)
