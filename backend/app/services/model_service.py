import mlflow.pyfunc
from config.settings import settings
import pandas as pd

class ModelService:
    def __init__(self):
        self.models = {}

    def load_model(self):
        model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        self.models["default"] = model

    def predict(self, text):
        model = self.models["default"]
        df = pd.DataFrame({"text": [text]})
        result = model.predict(df)

        return result.iloc[0]["prediction"]
