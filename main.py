"""
Sentinel Fraud Detection - Tier-1 FastAPI Gateway.
Triton inference, SHAP Rationale for High Risk, Prometheus metrics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"
THRESHOLD = 0.5

app = FastAPI(title="Sentinel Fraud Detection API", version="2.0.0")

feature_names: List[str] = []
triton_client = None
xai_explainer = None
drift_monitor = None

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
)
PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total predict requests (for QPS)",
)


class Rationale(BaseModel):
    top_features: List[Tuple[str, float]] = Field(
        ...,
        description="Top 3 contributing features (name, shap_value)",
    )
    risk_level: str = Field(..., description="Risk level e.g. High")


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Feature map for a single transaction (engineered features).",
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence_score: str
    rationale: Optional[Rationale] = None


@app.on_event("startup")
def load_artifacts() -> None:
    global feature_names, triton_client, xai_explainer, drift_monitor
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    import os
    triton_url = os.environ.get("TRITON_URL", "triton:8001")
    try:
        from src.triton_client import TritonClient
        triton_client = TritonClient(url=triton_url, model_name="sentinel_model")
    except Exception as e:
        triton_client = None
    try:
        from src.xai import SentinelExplainer
        xai_explainer = SentinelExplainer()
    except Exception:
        xai_explainer = None
    try:
        from monitoring.drift import StreamingDriftMonitor
        drift_monitor = StreamingDriftMonitor(buffer_size=10000)
    except Exception:
        drift_monitor = None


def _confidence_score(probability: float, threshold: float = THRESHOLD) -> str:
    distance = abs(probability - threshold)
    if distance >= 0.25:
        return "High"
    if distance >= 0.10:
        return "Medium"
    return "Low"


def _is_high_risk(probability: float, confidence: str) -> bool:
    return confidence == "High" and (probability >= THRESHOLD or probability > 0.75)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client not available")
    missing = [n for n in feature_names if n not in request.features]
    extra = [n for n in request.features if n not in feature_names]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        raise HTTPException(status_code=422, detail=f"Unexpected features: {extra[:10]}{'...' if len(extra) > 10 else ''}")

    ordered = [request.features[n] for n in feature_names]
    data = np.array([ordered], dtype=np.float32)

    start = time.perf_counter()
    try:
        probability = triton_client.predict(data)
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)
        PREDICT_REQUESTS.inc()
        if drift_monitor is not None:
            drift_monitor.add(data[0], feature_names)
            if drift_monitor.buffer_len() >= 1000:
                drift_monitor.flush()

    is_fraud = probability >= THRESHOLD
    confidence = _confidence_score(probability)
    rationale = None
    if _is_high_risk(probability, confidence) and xai_explainer is not None:
        try:
            top_feats = xai_explainer.compute_rationale(data, feature_names, top_k=3)
            rationale = Rationale(
                top_features=top_feats,
                risk_level="High",
            )
        except Exception:
            pass

    return PredictionResponse(
        fraud_probability=probability,
        is_fraud=is_fraud,
        confidence_score=confidence,
        rationale=rationale,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return generate_latest()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
