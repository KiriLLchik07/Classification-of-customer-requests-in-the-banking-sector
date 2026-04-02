from fastapi import APIRouter, HTTPException
import mlflow
import logging
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/model_info/{model_name}")
def model_info(model_name: str):
    """
    Возвращает информацию о модели из MLflow Registry
    """
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        versions = client.get_latest_versions(model_name)

        if not versions:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not founded in registry")

        result = []
        for v in versions:
            result.append({
                "version": v.version,
                "stage": v.current_stage,
                "description": v.description
            })

        return {
            "model_name": model_name,
            "versions": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info. Model={model_name}, error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
