from mlflow.pyfunc import PyFuncModel
import mlflow
import pandas as pd
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class ModelService:
    """
    Сервис для загрузки моделей из MLflow и инференса.
    """
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.models: dict[str, PyFuncModel] = {}

    def load_model(self, model_name: str, stage: str = "Production") -> PyFuncModel:
        """
        Загружает модель из MLflow Model Registry.
        Args:
            model_name: Имя модель в MLflow реестре.
            stage: Стадия модели.
        """
        model_uri = f"models:/{model_name}/{stage}"

        logger.info(f"Loading model {model_name} | stage={stage} | from {model_uri}")

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model | name={model_name} | error={str(e)}")
            raise

    def get_model(self, model_name: str) -> PyFuncModel:
        """Получает загруженную модель"""

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded! List of models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def predict(self, text: str, model_name: str) -> str:
        """
        Выполняет предсказание для одного текста.
        Args:
            text: Запрос пользователя.
            model_name: Название используемой для инференса модели.
        Returns: Предсказанный класс
        """
        
        model = self.get_model(model_name)
        df = pd.DataFrame({"text": [text]})
        result = model.predict(df)

        prediction = str(result.iloc[0, 0])

        logger.info(
            f"Prediction made | model={model_name} | text={text[:30]} | pred={prediction}"
        )

        return str(prediction)

    def list_models(self) -> list[str]:
        """Возвращает список загруженных моделей."""
        return list(self.models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Проверяет, загружена ли модель."""
        return model_name in self.models

model_service = ModelService()

def get_model_service():
    """Для получения сервиса. Используется для FastAPI"""
    return model_service
