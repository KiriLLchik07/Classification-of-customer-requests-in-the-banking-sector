from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Настройки проекта"""
    project_name: str = "Banking77 ML Service"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "Banking77_DistilBERT"
    model_alias: str = "production"
    device: str = "cpu"

    max_text_length: int = 512
    min_text_length: int = 3

    model_config = SettingsConfigDict(
        env_file=("config/environment.env", ".env"),
        extra="ignore",
        case_sensitive=False
    )
    
@lru_cache
def get_settings() -> Settings:
    """ Кэшированный инстанс настроек """
    return Settings()

settings = get_settings()
