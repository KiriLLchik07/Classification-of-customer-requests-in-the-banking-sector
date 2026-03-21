from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Настройки проекта"""
    project_name: str = "Banking77 ML Service"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    mlflow_tracking_uri: str = "http://mlflow:5000"
    model_name: str = "Banking77_Classifier"
    model_stage: str = "Production"
    device: str = "cpu"

    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    cache_total: int = 3600

    max_text_length: int = 512
    min_text_length: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
@lru_cache
def get_settings() -> Settings:
    """ Кэшированный инстанс настроек """
    return Settings()

settings = get_settings()
# (f"sqlite:///{(BASE_DIR / 'mlflow_data' / 'mlflow.db').as_posix()}")
