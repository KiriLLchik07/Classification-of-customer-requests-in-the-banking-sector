from pydantic import BaseModel, Field, field_validator
from config.settings import settings

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Текст запроса от клиента"
    )
    model_name: str = Field(
        default=settings.model_name,
        description="Название модели для инференса"
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return value.strip()
