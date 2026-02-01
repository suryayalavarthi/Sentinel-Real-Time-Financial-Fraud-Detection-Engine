from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "sentinel_fraud_model.json"
FEATURES_PATH = BASE_DIR.parent / "models" / "feature_names.json"
THRESHOLD = 0.5

app = FastAPI(title="Sentinel Fraud Detection API", version="1.0.0")

model: xgb.Booster | None = None
feature_names: List[str] = []


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Feature map for a single transaction (engineered features).",
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence_score: str


@app.on_event("startup")
def load_artifacts() -> None:
    global model, feature_names
    try:
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        model = booster
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    try:
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load feature names: {exc}") from exc


def _confidence_score(probability: float, threshold: float = THRESHOLD) -> str:
    distance = abs(probability - threshold)
    if distance >= 0.25:
        return "High"
    if distance >= 0.10:
        return "Medium"
    return "Low"


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None or not feature_names:
        raise HTTPException(status_code=503, detail="Model not loaded")

    missing = [name for name in feature_names if name not in request.features]
    extra = [name for name in request.features.keys() if name not in feature_names]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing[:10]}{'...' if len(missing) > 10 else ''}",
        )
    if extra:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected features: {extra[:10]}{'...' if len(extra) > 10 else ''}",
        )

    ordered = [request.features[name] for name in feature_names]
    data = np.array([ordered], dtype=np.float32)
    dmatrix = xgb.DMatrix(data, feature_names=feature_names)

    probability = float(model.predict(dmatrix)[0])
    is_fraud = probability >= THRESHOLD
    confidence = _confidence_score(probability)

    return PredictionResponse(
        fraud_probability=probability,
        is_fraud=is_fraud,
        confidence_score=confidence,
    )
