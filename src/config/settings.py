from pathlib import Path
from pydantic import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    project_name: str = "banking-intent-classifier"
    data_dir: Path = BASE_DIR / "data"
    model_dir: Path = BASE_DIR / "models"
    mlflow_tracking_uri: str = "http://localhost:5000"

    class Config:
        env_file = ".env"

settings = Settings()
