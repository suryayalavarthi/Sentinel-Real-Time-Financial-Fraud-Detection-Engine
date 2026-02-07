"""
SHAP-based Rationale for High-Risk transactions (regulatory explainability).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentinel_fraud_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"
TOP_K = 3
RISK_LEVEL = "High"


class SentinelExplainer:
    """
    SHAP TreeExplainer for XGBoost. Produces top-K feature attributions for Rationale.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_names_path: Path = FEATURES_PATH,
    ) -> None:
        self.model = joblib.load(model_path)
        with open(feature_names_path) as f:
            self.feature_names = json.load(f)
        self.explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation="tree_path_dependent",
        )

    def compute_rationale(
        self,
        feature_vector: np.ndarray,
        feature_names: List[str],
        top_k: int = TOP_K,
    ) -> List[Tuple[str, float]]:
        """
        Compute top-K contributing features by |SHAP value|.

        Args:
            feature_vector: Shape (1, n_features) float32.
            feature_names: Ordered feature names.
            top_k: Number of top features to return.

        Returns:
            List of (feature_name, shap_value) sorted by |shap_value| descending.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        shap_values = self.explainer.shap_values(feature_vector)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = shap_values
        indices = np.argsort(-np.abs(sv))[:top_k]
        return [(feature_names[i], float(sv[i])) for i in indices if i < len(feature_names)]
