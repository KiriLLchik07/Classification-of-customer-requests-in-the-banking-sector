from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from services.backend.app.api.v1.predict import router as predict_router
from services.backend.app.api.v1.models import router as models_router
from services.backend.app.api.v1.health import router as health_router
from services.backend.app.api.v1.model_info import router as model_info_router
from services.backend.app.api.v1.mlflow_models import router as mlflow_models_router
from services.backend.app.services.model_service import model_service
from services.backend.app.core.logger import setup_logging
from config.settings import settings
from prometheus_fastapi_instrumentator import PrometheusFastApiInstrumentator

logger = setup_logging(debug=settings.debug)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Жизненный цикл моделей
    """

    logger.info("Start application")

    try:
        model_service.load_model(settings.model_name, alias=settings.model_alias)
        logger.info(
            "Model loaded successfully. Model name: %s | alias=%s",
            settings.model_name,
            settings.model_alias,
        )
    except Exception as e:
        logger.warning(
            "Startup model load skipped. Model=%s | alias=%s | error=%s",
            settings.model_name,
            settings.model_alias,
            str(e),
        )

    yield

    logger.info("Shutdowning application")

app = FastAPI(title=settings.project_name, version="1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PrometheusFastApiInstrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(health_router, prefix=settings.api_v1_prefix, tags=["Health"])
app.include_router(predict_router, prefix=settings.api_v1_prefix, tags=["Prediction"])
app.include_router(models_router, prefix=settings.api_v1_prefix, tags=["Models"])
app.include_router(model_info_router, prefix=settings.api_v1_prefix, tags=["Model info"])
app.include_router(mlflow_models_router, prefix=settings.api_v1_prefix, tags=["MLflow models"])

@app.get("/")
def root():
    """Коревой эндпоинт"""
    return {
        "service": settings.project_name,
        "version": "1.0",
        "docs": "/docs"
    }
