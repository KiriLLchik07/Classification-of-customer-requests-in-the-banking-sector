from mlflow.pyfunc import PyFuncModel
import mlflow
import pandas as pd
from functools import lru_cache
import logging

logger = logging.getLogger("banking77")

class ModelService:
    def __init__(self):
        self.models: dict[str, PyFuncModel] = {}

    def load_model(self, model_name: str):
        model_uri = f"models:/{model_name}/Production"

        logger.info(f"Loading model {model_name} from {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        self.models[model_name] = model

        logger.info(f"Model {model_name} loaded")

    @lru_cache(maxsize=1000)
    def predict(self, text: str, model_name: str) -> int:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]
        df = pd.DataFrame({"text": [text]})
        result = model.predict(df)
        prediction = int(result[0])

        logger.info(
            f"Prediction made | model={model_name} | text={text[:30]} | pred={prediction}"
        )

        return prediction

    def list_models(self):
        return list(self.models.keys())

model_service = ModelService()

def get_model_service():
    return model_service
