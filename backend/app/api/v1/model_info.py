from fastapi import APIRouter
import mlflow

router = APIRouter()

@router.get("/model_info/{model_name}")
def model_info(model_name: str):
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name)
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
