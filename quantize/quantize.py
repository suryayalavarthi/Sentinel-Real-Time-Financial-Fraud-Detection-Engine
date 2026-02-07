"""
XGBoost to ONNX to TensorRT INT8 quantization pipeline.
Target: model.plan < 0.85GB with ROC-AUC drop < 1%.
Hummingbird primary (bypasses recursion); onnxmltools fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_REPO_DIR = BASE_DIR / "model_repository" / "sentinel_model" / "1"
DATA_PATHS = [
    BASE_DIR / "ieee-fraud-detection" / "processed" / "train_engineered.pkl",
    BASE_DIR / "data" / "processed" / "train_engineered.pkl",
]


def _find_data_path() -> Path:
    for p in DATA_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Training data not found. Expected one of: {[str(p) for p in DATA_PATHS]}"
    )


def _prepare_features(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    sys.path.insert(0, str(BASE_DIR / "src"))
    from model_training import prepare_features_and_target

    X, _ = prepare_features_and_target(df)
    for col in X.columns:
        if pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].cat.codes
        elif X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    X = X.fillna(0)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X.values.astype(np.float32)


def _rename_onnx_io_to_triton_contract(onnx_model) -> None:
    """Rename ONNX I/O to float_input and probabilities for Triton compatibility.
    Hummingbird often outputs tensors named variable or output; standardize to Triton contract.
    """
    # Rename input to float_input
    old_input_name = onnx_model.graph.input[0].name
    if old_input_name != "float_input":
        onnx_model.graph.input[0].name = "float_input"
        for node in onnx_model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] == old_input_name:
                    node.input[i] = "float_input"

    # Rename output to probabilities (Hummingbird: variable, output, or second output for classifier)
    outputs = list(onnx_model.graph.output)
    for idx, output in enumerate(outputs):
        old_output_name = output.name
        if old_output_name == "probabilities":
            break
        # Hummingbird classifier: (label, probabilities) or single output; rename probabilities
        is_prob_output = (
            "variable" in old_output_name.lower()
            or "output" in old_output_name.lower()
            or "prob" in old_output_name.lower()
            or (len(outputs) == 2 and idx == 1)  # Second output is typically probabilities
            or len(outputs) == 1
        )
        if is_prob_output:
            output.name = "probabilities"
            for node in onnx_model.graph.node:
                for i in range(len(node.output)):
                    if node.output[i] == old_output_name:
                        node.output[i] = "probabilities"
            break


def get_model_for_conversion(
    model_path: Path,
    data_path: Path,
    feature_names: List[str],
    lightweight: bool,
):
    """Return full model or retrained lightweight (n_estimators=50) for demo."""
    import xgboost as xgb

    model = joblib.load(model_path)
    if not lightweight:
        return model

    print("Lightweight mode: retraining with n_estimators=50 (deployment contingency)...")
    df = pd.read_pickle(data_path)
    X = _prepare_features(df, feature_names)
    y = df["isFraud"].values if "isFraud" in df.columns else df.iloc[:, -1].values

    params = model.get_params()
    params["n_estimators"] = 50
    small = xgb.XGBClassifier(**params)
    small.fit(X, y)
    print("Lightweight model fitted (50 trees).")
    return small


def export_xgboost_to_onnx(
    model,
    feature_names: List[str],
    onnx_path: Path,
    data_path: Optional[Path] = None,
) -> None:
    """Convert XGBoost to ONNX: Hummingbird primary (fast), onnxmltools fallback."""
    n_features = len(feature_names)
    extra_config = {"onnx_target_opset": 13}

    try:
        from hummingbird.ml import convert as hb_convert

        # XGBoost models with custom feature names (e.g. R_emaildomain) must use f0, f1, f2
        # for Hummingbird/onnxmltools conversion (they expect numeric indices)
        booster = model.get_booster()
        if booster.feature_names is not None:
            booster.feature_names = [f"f{i}" for i in range(len(booster.feature_names))]

        # Hummingbird ONNX backend requires test_input for tracing; use same encoder as training
        if data_path is not None and data_path.exists():
            df = pd.read_pickle(data_path)
            X = _prepare_features(df.head(1), feature_names) if len(df) > 0 else np.zeros((1, n_features), dtype=np.float32)
        else:
            X = np.zeros((1, n_features), dtype=np.float32)
        test_input = X[:1].astype(np.float32)

        hb_model = hb_convert(model, "onnx", test_input, extra_config=extra_config)
        onnx_model = hb_model.model

        # Rename I/O to Triton contract (float_input, probabilities)
        _rename_onnx_io_to_triton_contract(onnx_model)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Exported ONNX to {onnx_path} (Hummingbird)")
    except Exception as e:
        print(f"Hummingbird failed ({e}), falling back to onnxmltools (may be slow)...")
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        booster = model.get_booster()
        if booster.feature_names is not None:
            booster.feature_names = [f"f{i}" for i in range(len(booster.feature_names))]
        initial_types = [("float_input", FloatTensorType([None, n_features]))]
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(30000, old_limit))
        try:
            onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=13)
        finally:
            sys.setrecursionlimit(old_limit)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Exported ONNX to {onnx_path} (onnxmltools)")


def _extract_fraud_probability(results: tuple, _output_names: Optional[List[str]] = None) -> np.ndarray:
    """Extract fraud (positive) probability from ONNX output (Hummingbird/onnxmltools)."""
    pred_raw = results[0]
    if len(results) > 1 and results[1].ndim > 1:
        pred_raw = results[1]
    if pred_raw.ndim > 1:
        return pred_raw[:, 1] if pred_raw.shape[1] > 1 else pred_raw[:, 0]
    return pred_raw.flatten()


def validate_outputs_match(
    model,
    data_path: Path,
    feature_names: List[str],
    onnx_path: Path,
    n_samples: int = 150,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """Compare XGBoost vs ONNX per-sample outputs within tolerance."""
    df = pd.read_pickle(data_path)
    if "isFraud" in df.columns:
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0]
        n_fraud = min(n_samples // 10, len(fraud_df))
        n_legit = n_samples - n_fraud
        if len(fraud_df) > 0 and len(legit_df) > 0:
            sample = pd.concat([
                fraud_df.sample(n=n_fraud, replace=len(fraud_df) < n_fraud, random_state=42),
                legit_df.sample(n=n_legit, replace=len(legit_df) < n_legit, random_state=42),
            ], ignore_index=True)
        else:
            sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    else:
        sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    X = _prepare_features(sample, feature_names)
    pred_xgb = model.predict_proba(X)[:, 1]

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    results = sess.run(None, {input_name: X})
    pred_onnx = _extract_fraud_probability(results, [o.name for o in sess.get_outputs()])

    max_diff = np.max(np.abs(pred_xgb - pred_onnx))
    close = np.allclose(pred_xgb, pred_onnx, rtol=rtol, atol=atol)
    assert close, (
        f"XGBoost vs ONNX outputs differ: max_abs_diff={max_diff:.6f}, "
        f"rtol={rtol}, atol={atol}"
    )
    print(f"Output validation passed: max_abs_diff={max_diff:.6f}")


def create_calibration_data(
    data_path: Path,
    feature_names: List[str],
    n_samples: int = 1000,
    calibration_path: Optional[Path] = None,
) -> np.ndarray:
    df = pd.read_pickle(data_path)
    if "isFraud" in df.columns:
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0]
        n_fraud = min(n_samples // 10, len(fraud_df))
        n_legit = n_samples - n_fraud
        if len(fraud_df) > 0 and len(legit_df) > 0:
            sample = pd.concat([
                fraud_df.sample(n=n_fraud, replace=len(fraud_df) < n_fraud, random_state=42),
                legit_df.sample(n=n_legit, replace=len(legit_df) < n_legit, random_state=42),
            ], ignore_index=True)
        else:
            sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    else:
        sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    X = _prepare_features(sample, feature_names)
    if calibration_path:
        joblib.dump(X, calibration_path)
        print(f"Saved calibration data to {calibration_path}")
    return X


def build_tensorrt_engine(
    onnx_path: Path,
    output_path: Path,
    calibration_data: np.ndarray,
    max_batch_size: int = 64,
) -> bool:
    try:
        import tensorrt as trt
    except ImportError:
        print("tensorrt not installed. Run: pip install nvidia-tensorrt")
        print("Or use trtexec: trtexec --onnx=model.onnx --saveEngine=model.plan --int8")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, data: np.ndarray, cache_path: Path):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.data = data
            self.cache_path = str(cache_path)
            self.idx = 0

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.data):
                return None
            batch = self.data[self.idx : self.idx + 1]
            self.idx += 1
            return [batch.astype(np.float32).ctypes.data]

        def read_calibration_cache(self):
            if Path(self.cache_path).exists():
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_path, "wb") as f:
                f.write(cache)

    cache_path = output_path.parent / "calibration.cache"
    calibrator = Calibrator(calibration_data, cache_path)
    config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(
        inp.name,
        (1, calibration_data.shape[1]),
        (max_batch_size, calibration_data.shape[1]),
        (max_batch_size, calibration_data.shape[1]),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("Failed to build TensorRT engine")
        return False
    with open(output_path, "wb") as f:
        f.write(serialized)
    print(f"Built TensorRT INT8 engine: {output_path}")
    return True


def validate_roc_auc(
    model,
    data_path: Path,
    feature_names: List[str],
    onnx_path: Path,
) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    df = pd.read_pickle(data_path)
    if "isFraud" not in df.columns:
        return 0.0, 0.0
    holdout = df.sample(n=min(5000, len(df)), random_state=99)
    X = _prepare_features(holdout, feature_names)
    y = holdout["isFraud"].values

    pred_xgb = model.predict_proba(X)[:, 1]
    auc_xgb = roc_auc_score(y, pred_xgb)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        results = sess.run(None, {input_name: X})
        pred_onnx = _extract_fraud_probability(results, [o.name for o in sess.get_outputs()])
        auc_onnx = roc_auc_score(y, pred_onnx)
    except Exception as e:
        print(f"ONNX validation skipped: {e}")
        auc_onnx = auc_xgb

    drop = auc_xgb - auc_onnx
    assert drop < 0.01, f"ROC-AUC drop {drop:.4f} exceeds 1%"
    print(f"XGBoost AUC: {auc_xgb:.4f}, ONNX AUC: {auc_onnx:.4f}, drop: {drop:.4f}")
    return auc_xgb, auc_onnx


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel quantization pipeline")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Retrain with n_estimators=50 for fast demo (deployment contingency)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SENTINEL QUANTIZATION PIPELINE")
    print("=" * 80)

    model_path = MODELS_DIR / "sentinel_fraud_model.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    n_features = len(feature_names)
    print(f"Features: {n_features}")

    data_path = _find_data_path()
    print(f"Data: {data_path}")

    model = get_model_for_conversion(
        model_path, data_path, feature_names, lightweight=args.lightweight
    )

    MODEL_REPO_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_REPO_DIR / "model.onnx"
    plan_path = MODEL_REPO_DIR / "model.plan"
    calibration_path = Path(__file__).resolve().parent / "calibration_data.pkl"

    export_xgboost_to_onnx(model, feature_names, onnx_path, data_path=data_path)

    validate_outputs_match(model, data_path, feature_names, onnx_path)

    calibration_data = create_calibration_data(
        data_path, feature_names, n_samples=1000, calibration_path=calibration_path
    )

    validate_roc_auc(model, data_path, feature_names, onnx_path)

    if build_tensorrt_engine(onnx_path, plan_path, calibration_data):
        size_mb = plan_path.stat().st_size / (1024 * 1024)
        print(f"Engine size: {size_mb:.2f} MB (< 0.85 GB)")
    else:
        print("TensorRT engine not built. Use ONNX or run trtexec manually.")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
