from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    mlflow_tracking_uri: str = "http://localhost:5000"

    model_name: str = "Banking77_Classifier"

    model_stage: str = "Production"


settings = Settings()
