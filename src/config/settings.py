from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    project_name: str = "Classification-of-customer-requests-in-the-banking-sector"

    mlflow_dir: Path = BASE_DIR / "mlflow_config"
    mlflow_tracking_uri: str = (f"sqlite:///{(BASE_DIR / 'mlflow_config' / 'mlflow.db').as_posix()}")

    class Config:
        env_file = ".env"

settings = Settings()
