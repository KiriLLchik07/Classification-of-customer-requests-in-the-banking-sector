from fastapi import APIRouter, Depends
from app.services.model_service import get_model_service, ModelService

router = APIRouter()

@router.get("/models")
def get_models(model_service: ModelService = Depends(get_model_service)):
    """Возвращает список загруженных моделей"""
    models = model_service.list_models()
    return {"models": models}
