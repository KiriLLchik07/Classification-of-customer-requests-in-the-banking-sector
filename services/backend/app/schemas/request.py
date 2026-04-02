from pydantic import BaseModel, Field, field_validator
from config.settings import settings

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="The text of the request from the client"
    )
    model_name: str = Field(
        default=settings.model_name,
        description="The name of the model for inference"
    )
    model_stage: str = Field(
        default=settings.model_stage,
        description="The stage of the model in MLflow Registry"
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return value.strip()
