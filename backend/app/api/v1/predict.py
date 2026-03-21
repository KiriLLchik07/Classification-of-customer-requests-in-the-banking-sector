import logging
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.model_service import ModelService, get_model_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, model_service: ModelService = Depends(get_model_service)):
    """
    Выполняет предсказание модели(классификация запроса).
    Args:
        request: Текст запроса и имя модели.
        model_service: Сервис моделей.
    Returns:
        Предсказанный класс модели
    """
    logger.info(f"Request received | model={request.model_name}")

    if not model_service.is_model_loaded(request.model_name):
        try:
            logger.info(f"Model not loaded. Try to load model {request.model_name}")
            model_service.load_model(request.model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not available. Error: {str(e)}")

    try:
        prediction = model_service.predict(model_name=request.model_name, text=request.text)
    except Exception as e:
        logger.error(f"Prediction failed: model={request.model_name}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction_failed. Error: {str(e)}")

    return PredictResponse(prediction=prediction, model_name=request.model_name)
