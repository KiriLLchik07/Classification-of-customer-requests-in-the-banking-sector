import mlflow
from functools import lru_cache
from app.core.config import settings

@lru_cache
def get_model():

    model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    return model
