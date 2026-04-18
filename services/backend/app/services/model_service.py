from mlflow.pyfunc import PyFuncModel
import mlflow
import pandas as pd
import logging
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

class ModelService:
    """
    Сервис для загрузки моделей из MLflow и инференса.
    """
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.models: dict[str, PyFuncModel] = {}
        self.model_aliases: dict[str, str] = {}
        self.code_to_label: dict[int, str] = {}
        self.label_to_code: dict[str, int] = {}
        self._load_label_mapping()

    def _load_label_mapping(self) -> None:
        project_root = Path(__file__).resolve().parents[4]
        train_df_path = project_root / "data" / "processed" / "train_df.csv"

        try:
            df = pd.read_csv(train_df_path)
            deduplicated = df[["label", "intent"]].drop_duplicates()
            self.code_to_label = {int(row["label"]): str(row["intent"]) for _, row in deduplicated.iterrows()}
            self.label_to_code = {label.strip().lower(): code for code, label in self.code_to_label.items()}
            logger.info("Label mapping loaded from %s", train_df_path)
            return
        except Exception as e:
            logger.warning("Could not load label mapping from %s: %s", train_df_path, str(e))

        logger.warning("Label mapping was not loaded. Prediction code/label resolution may be limited.")

    def load_model(self, model_name: str, alias: str = "production") -> PyFuncModel:
        """
        Загружает модель из MLflow Model Registry.
        Args:
            model_name: Имя модель в MLflow реестре.
            alias: Состояние модели.
        """
        model_uri = f"models:/{model_name}@{alias}"

        logger.info(f"Loading model {model_name} | alias={alias} | from {model_uri}")

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            self.models[model_name] = model
            self.model_aliases[model_name] = alias
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
        return prediction

    def resolve_prediction(self, raw_prediction: str | int | float) -> tuple[str, int | None]:
        prediction_str = str(raw_prediction).strip()

        try:
            prediction_code = int(float(prediction_str))
            prediction_label = self.code_to_label.get(prediction_code, prediction_str)
            return prediction_label, prediction_code
        except ValueError:
            pass

        prediction_code = self.label_to_code.get(prediction_str.lower())
        if prediction_code is None:
            return prediction_str, None

        prediction_label = self.code_to_label.get(prediction_code, prediction_str)
        return prediction_label, prediction_code

    def list_models(self) -> list[str]:
        """Возвращает список загруженных моделей."""
        return list(self.models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Проверяет, загружена ли модель."""
        return model_name in self.models

    def get_loaded_alias(self, model_name: str) -> str | None:
        return self.model_aliases.get(model_name)

model_service = ModelService()

def get_model_service():
    """Для получения сервиса. Используется для FastAPI"""
    return model_service
