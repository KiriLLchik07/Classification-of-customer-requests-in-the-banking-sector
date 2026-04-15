import logging
from fastapi import APIRouter, Depends, HTTPException
from services.backend.app.schemas.request import PredictRequest
from services.backend.app.schemas.response import PredictResponse
from services.backend.app.services.model_service import ModelService, get_model_service

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

    loaded_alias = model_service.get_loaded_alias(request.model_name)
    model_needs_loading = (
        (not model_service.is_model_loaded(request.model_name))
        or (loaded_alias is not None and loaded_alias != request.model_alias)
        or (loaded_alias is None)
    )

    if model_needs_loading:
        try:
            logger.info(
                "Model load required. model=%s | requested_alias=%s | loaded_alias=%s",
                request.model_name,
                request.model_alias,
                loaded_alias,
            )
            model_service.load_model(request.model_name, alias=request.model_alias)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not available. Error: {str(e)}")

    try:
        prediction = model_service.predict(model_name=request.model_name, text=request.text)
    except Exception as e:
        logger.error(f"Prediction failed: model={request.model_name}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction_failed. Error: {str(e)}")

    return PredictResponse(prediction=prediction, model_name=request.model_name)
