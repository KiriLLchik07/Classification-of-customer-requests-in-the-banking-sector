from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.v1.predict import router as predict_router
from app.services.model_service import ModelService

model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):

    model_service.load_model("Banking77_DistilBERT")
    model_service.load_model("Banking77_LSTM")
    model_service.load_model("Banking77_LogReg")

    yield

app = FastAPI(
    title="Banking77 Intent Classifier",
    version="1.0",
    lifespan=lifespan
)


app.include_router(
    predict_router,
    prefix="/api/v1"
)
