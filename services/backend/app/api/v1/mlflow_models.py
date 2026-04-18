from fastapi import APIRouter, HTTPException
import mlflow
import logging
from services.backend.app.schemas.response import MlflowModelsResponse
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/mlflow_models", response_model=MlflowModelsResponse)
def get_mlflow_models():
    """
    Получает список всех зарегистрированных модель в MLflow
    """
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        models = client.search_registered_models()
        model_names = [model.name for model in models]
        return MlflowModelsResponse(model_names=model_names)
    except Exception as e:
        logger.error("Failed to fetch models from MLflow Registry: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch models from MLflow Registry")
