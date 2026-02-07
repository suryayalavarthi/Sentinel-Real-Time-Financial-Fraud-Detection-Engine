"""
PSI and KL divergence for feature drift detection in streaming context.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prometheus_client import Gauge

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATHS = [
    BASE_DIR / "ieee-fraud-detection" / "processed" / "train_engineered.pkl",
    BASE_DIR / "data" / "processed" / "train_engineered.pkl",
]

DRIFT_SCORE_PSI = Gauge(
    "drift_score_psi",
    "Max Population Stability Index across monitored features",
)


def calculate_psi(
    reference: np.ndarray,
    production: np.ndarray,
    bins: int = 10,
) -> float:
    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]
    if len(reference) == 0 or len(production) == 0:
        return 0.0
    _, bin_edges = np.histogram(reference, bins=bins)
    min_val = min(float(reference.min()), float(production.min()))
    max_val = max(float(reference.max()), float(production.max()))
    bin_edges[0] = min_val - 1e-6
    bin_edges[-1] = max_val + 1e-6
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)
    ref_percents = ref_counts / len(reference)
    prod_percents = prod_counts / len(production)
    ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
    prod_percents = np.where(prod_percents == 0, 0.0001, prod_percents)
    psi = np.sum((prod_percents - ref_percents) * np.log(prod_percents / ref_percents))
    return float(psi)


def calculate_kl_divergence(
    reference: np.ndarray,
    production: np.ndarray,
    bins: int = 10,
) -> float:
    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]
    if len(reference) == 0 or len(production) == 0:
        return 0.0
    _, bin_edges = np.histogram(reference, bins=bins)
    min_val = min(float(reference.min()), float(production.min()))
    max_val = max(float(reference.max()), float(production.max()))
    bin_edges[0] = min_val - 1e-6
    bin_edges[-1] = max_val + 1e-6
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)
    ref_probs = ref_counts / len(reference)
    prod_probs = prod_counts / len(production)
    ref_probs = np.where(ref_probs == 0, 1e-10, ref_probs)
    prod_probs = np.where(prod_probs == 0, 1e-10, prod_probs)
    kl_div = np.sum(prod_probs * np.log(prod_probs / ref_probs))
    return float(kl_div)


def _find_data_path() -> Path:
    for p in DATA_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"Training data not found. Expected one of: {[str(p) for p in DATA_PATHS]}")


class StreamingDriftMonitor:
    """
    Ring buffer of feature vectors for streaming PSI drift detection.
    Every N requests or on flush, computes PSI per monitored feature and returns max_psi.
    """

    DEFAULT_FEATURES = [
        "TransactionAmt",
        "uid_TransactionFreq_24h",
        "uid_TransactionAmt_mean_30d",
        "Amt_to_Mean_Ratio",
        "card1_freq",
    ]

    def __init__(
        self,
        buffer_size: int = 10_000,
        features_to_monitor: Optional[List[str]] = None,
        reference_data_path: Optional[Path] = None,
        bins: int = 10,
    ) -> None:
        self.buffer_size = buffer_size
        self.features_to_monitor = features_to_monitor or self.DEFAULT_FEATURES
        self.bins = bins
        self._buffer: deque = deque(maxlen=buffer_size)
        self._feature_indices: Dict[str, int] = {}
        self._reference_arrays: Dict[str, np.ndarray] = {}
        self._reference_bins_path = Path(__file__).resolve().parent / "reference_bins.pkl"
        self._load_reference(reference_data_path)

    def _load_reference(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = _find_data_path()
        sys.path.insert(0, str(BASE_DIR / "src"))
        from model_training import prepare_features_and_target

        df = pd.read_pickle(path)
        X_ref, _ = prepare_features_and_target(df)
        for col in X_ref.columns:
            if pd.api.types.is_categorical_dtype(X_ref[col]):
                X_ref[col] = X_ref[col].cat.codes
            elif X_ref[col].dtype == object:
                X_ref[col] = X_ref[col].astype("category").cat.codes
        X_ref = X_ref.fillna(0)
        for i, col in enumerate(X_ref.columns):
            self._feature_indices[col] = i
        for feat in self.features_to_monitor:
            if feat in X_ref.columns:
                self._reference_arrays[feat] = X_ref[feat].values.astype(np.float64)

    def add(self, feature_vector: np.ndarray, feature_names: List[str]) -> None:
        self._buffer.append((feature_vector.copy(), feature_names))

    def buffer_len(self) -> int:
        return len(self._buffer)

    def flush(self) -> float:
        if len(self._buffer) < 2:
            return 0.0
        max_psi = 0.0
        for feat in self.features_to_monitor:
            if feat not in self._reference_arrays:
                continue
            ref_arr = self._reference_arrays[feat]
            prod_values = []
            for vec, names in self._buffer:
                if feat in names:
                    idx = names.index(feat)
                    prod_values.append(float(vec[idx]))
            if not prod_values:
                continue
            prod_arr = np.array(prod_values, dtype=np.float64)
            psi = calculate_psi(ref_arr, prod_arr, bins=self.bins)
            max_psi = max(max_psi, psi)
        DRIFT_SCORE_PSI.set(max_psi)
        return max_psi
