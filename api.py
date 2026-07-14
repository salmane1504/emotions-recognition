from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from main import LABELS, load_model, predict, DEFAULT_THRESHOLD, MODEL_DIR


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64, description="List of texts to classify")
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Probability threshold")


class PredictResponse(BaseModel):
    predictions: list[dict[str, float]]


models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = _get_device()
    models["device"] = device
    models["model"] = load_model(device)
    models["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    yield
    models.clear()


def _get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


app = FastAPI(
    title="GoEmotions API",
    description="Multi-label emotion classification using fine-tuned RoBERTa",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict_emotions(request: PredictRequest):
    results = predict(
        models["model"],
        models["tokenizer"],
        request.texts,
        models["device"],
        request.threshold,
    )
    return PredictResponse(predictions=results)


@app.get("/health")
async def health():
    return {"status": "ok", "labels": LABELS}
