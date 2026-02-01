# 🛡️ Sentinel - High-Frequency Fraud Detection System

A production-ready, memory-optimized fraud detection pipeline built for large-scale transaction data under strict resource constraints.

## 🎯 Executive Summary

**Sentinel** delivers time-series fraud detection with a reproducible pipeline, containerized inference, and continuous monitoring. The project is organized for production use with `./src` for core pipeline code, `./data/processed` for engineered datasets, and `./models` for model artifacts.

## 🏗️ System Architecture

### Architecture Overview
The system is split into two cooperating services: a FastAPI inference service and a monitoring sidecar. The FastAPI service loads the trained model and feature list from `./models` and serves `/predict` requests from engineered inputs. The monitoring sidecar runs in the same Docker Compose stack, mounts `./models` and `./data/processed`, and continuously evaluates drift and performance using the same feature definitions. This separation keeps real-time inference responsive while monitoring runs asynchronously on batches.

### Feature Pipeline
All feature engineering is strictly chronological. Transactions are sorted by `TransactionDT`, and data splits follow time order to prevent future leakage. The `uid_TransactionFreq_24h` feature is computed per user by counting the number of prior transactions in the last 24 hours (rolling 86,400-second window) for that user, capturing bursty behavior without using any future data.

### Monitoring Logic
The sidecar uses Population Stability Index (PSI) to compare the distribution of key features in a production batch against the reference training distribution from `./data/processed/train_engineered.pkl`. PSI surfaces “silent” model decay where accuracy drops without obvious label shifts, allowing early detection of drift before it becomes a performance incident.

### Inference
The FastAPI service targets low-latency predictions. The Live Smoke Test against `http://localhost:8000/predict` confirmed sub-100ms response time (≈6–8 ms average on localhost), validating end-to-end inference speed with the current `./models` artifacts.

## ✅ Production Verification

**Live API Response (FastAPI `/predict`)**
```json
{
  "confidence_score": "High",
  "fraud_probability": 0.9999977350234985,
  "is_fraud": true
}
```

## 📁 Project Structure

```
sentinel/
├── src/                      # Core pipeline + API
├── scripts/                  # Monitoring, interpretability, utilities
├── data/
│   ├── raw/                  # Original CSVs
│   └── processed/            # Engineered .pkl outputs
├── models/                   # Trained model artifacts
├── logs/                     # Drift + monitoring outputs
├── reports/                  # Plots and reports
└── docs/README.md            # Full technical documentation
```

## 📚 Full Documentation

For detailed pipelines, benchmarks, and evaluation notes, see `docs/README.md`.
