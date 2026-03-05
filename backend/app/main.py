from fastapi import FastAPI
from app.api.v1.predict import router as predict_router

app = FastAPI(
    title="Banking77 Intent Classifier",
    version="1.0"
)

app.include_router(
    predict_router,
    prefix="/api/v1"
)
