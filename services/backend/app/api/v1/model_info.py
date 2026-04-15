from fastapi import APIRouter, HTTPException
import mlflow
import logging
from services.backend.app.schemas.response import ModelInfoResponse, ModelInfoVersion
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/model_info/{model_name}", response_model=ModelInfoResponse)
def model_info(model_name: str):
    """
    Возвращает информацию о модели из MLflow Registry
    """
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        requested_alias = "production"
        version = client.get_model_version_by_alias(name=model_name, alias=requested_alias)

        aliases = list(getattr(version, "aliases", []) or [requested_alias])
        return ModelInfoResponse(
            model_name=model_name,
            versions=[
                ModelInfoVersion(
                    version=str(version.version),
                    alias=requested_alias,
                    aliases=aliases,
                    description=version.description,
                )
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' with alias 'production' not found in registry",
            )
        logger.error(f"Failed to get model info. Model={model_name}, error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
