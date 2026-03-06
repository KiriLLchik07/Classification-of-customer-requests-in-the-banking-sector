from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_name: str = "Banking77 ML Service"

    mlflow_tracking_uri: str = "http://localhost:5000"
    # (f"sqlite:///{(BASE_DIR / 'mlflow_config' / 'mlflow.db').as_posix()}")

    model_name: str = "Banking77_Classifier"
    model_stage: str = "Production"
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()
