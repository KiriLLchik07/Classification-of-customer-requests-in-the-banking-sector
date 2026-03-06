from fastapi import APIRouter
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.model_service import ModelService

router = APIRouter()

model_service = ModelService()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    prediction = model_service.predict(request.text)

    return PredictResponse(prediction=prediction)
