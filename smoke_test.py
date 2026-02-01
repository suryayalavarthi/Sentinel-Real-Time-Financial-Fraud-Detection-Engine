import json
import sys
import time

import pandas as pd
import requests


API_URL = "http://localhost:8000/predict"
SAMPLE_PATH = "./data/processed/train_engineered.pkl"
FEATURES_PATH = "./models/feature_names.json"


def main() -> int:
    df = pd.read_pickle(SAMPLE_PATH)
    if df.empty:
        print("Sample dataset is empty.")
        return 1

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    sample_df = df.loc[:, feature_names].head(1).copy()
    for col in sample_df.columns:
        if isinstance(sample_df[col].dtype, pd.CategoricalDtype):
            sample_df[col] = sample_df[col].cat.codes
        elif sample_df[col].dtype == object:
            sample_df[col] = sample_df[col].astype("category").cat.codes

    sample_df = sample_df.fillna(0)
    sample = sample_df.iloc[0].to_dict()

    payload = {"features": sample}
    # Warm-up request
    requests.post(API_URL, json=payload, timeout=30)

    latencies_ms = []
    response = None
    for _ in range(5):
        start = time.perf_counter()
        response = requests.post(API_URL, json=payload, timeout=30)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    print(f"Status code: {response.status_code}")
    print(
        "Latency (ms): "
        + ", ".join(f"{latency:.2f}" for latency in latencies_ms)
        + f" | avg={sum(latencies_ms)/len(latencies_ms):.2f}"
    )
    try:
        response_json = response.json()
    except json.JSONDecodeError:
        print("Response was not valid JSON.")
        print(response.text)
        return 1

    print(json.dumps(response_json, indent=2, sort_keys=True))

    if response.status_code != 200:
        print("Smoke test failed: non-200 status code.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
