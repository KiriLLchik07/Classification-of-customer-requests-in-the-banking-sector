from fastapi import APIRouter, Depends

from app.schemas.request import PredictRequest
from app.schemas.responce import PredictResponse
from app.dependencies.model_loader import get_model
from app.services.inference_servic import InferenceService

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, model=Depends(get_model)):
    service = InferenceService(model)
    pred = service.predict(request.text)

    return PredictResponse(prediction=int(pred))
