import logging
from fastapi import APIRouter, Depends
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.model_service import ModelService, get_model_service

logger = logging.getLogger("banking77")
router = APIRouter()
model_service = ModelService()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, model_service: ModelService = Depends(get_model_service)):
    logger.info(f"Request received | model={request.model_name}")
    prediction = model_service.predict(model_name=request.model_name, text=request.text)

    return PredictResponse(prediction=prediction)
