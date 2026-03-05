from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=512
    )
