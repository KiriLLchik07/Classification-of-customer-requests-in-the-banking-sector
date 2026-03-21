from fastapi import APIRouter, Depends
import logging
from app.schemas.response import HealthResponce
from app.services.model_service import ModelService, get_model_service
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponce)
def health(model_service: ModelService = Depends(get_model_service)):
    """Эндпоинт для проверки состояния сервиса."""
    
    models_loaded = model_service.list_models()
    status = "Ok" if models_loaded else "UnHealth"

    if status == "UnHealth":
        logger.warning("Health check failed, no models loaded")
    
    return HealthResponce(
        status=status,
        service=settings.project_name,
        version="1.0",
        models_loaded=models_loaded
    )
