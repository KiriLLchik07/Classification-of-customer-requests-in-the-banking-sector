from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.v1.predict import router as predict_router
from app.api.v1.models import router as models_router
from app.api.v1.health import router as health_router
from app.api.v1.model_info import router as model_info_router

from app.services.model_service import model_service
from app.core.logger import setup_logging
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
        model_service.load_model(settings.model_name)
        logger.info(f"Model loaded successfully. Model name: {settings.model_name}")
    except Exception as e:
        logger.error(f"Failed to load model. Error: {str(e)}")

    yield

    logger.info("Shutdowning application")

app = FastAPI(title=settings.project_name, version="1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PrometheusFastApiInstrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(health_router, prefix=settings.api_v1_prefix, tags=["Health"])
app.include_router(predict_router, prefix=settings.api_v1_prefix, tags=["Prediction"])
app.include_router(models_router, prefix=settings.api_v1_prefix, tags=["Models"])
app.include_router(model_info_router, prefix=settings.api_v1_prefix, tags=["Model info"])

@app.get("/")
def root():
    """Коревой эндпоинт"""
    return {
        "service": settings.project_name,
        "version": "1.0",
        "docs": "/docs"
    }
