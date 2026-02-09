#!/usr/bin/env python3
"""
Generate a minimal ONNX model for CI smoke tests.
Loads feature count from models/feature_names.json to match main.py schema.
"""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"


def main() -> None:
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        raise SystemExit(
            "onnx required for CI model generation. Run: pip install onnx"
        )

    # Load feature count from same JSON as main.py (single source of truth)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    n_features = len(feature_names)

    # Triton config: input float_input [n_features], output probabilities [2]
    input_name = "float_input"
    output_name = "probabilities"
    n_classes = 2

    # Input: [batch, n_features], Output: [batch, 2]
    input_tensor = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [-1, n_features]
    )
    output_tensor = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [-1, n_classes]
    )

    # Gemm: Y = X @ W + B (simple linear layer)
    w = np.random.randn(n_features, n_classes).astype(np.float32) * 0.01
    b = np.array([0.0, 0.0], dtype=np.float32)

    w_init = helper.make_tensor("W", TensorProto.FLOAT, [n_features, n_classes], w.flatten().tolist())
    b_init = helper.make_tensor("B", TensorProto.FLOAT, [n_classes], b.tolist())

    gemm_out = "gemm_out"
    gemm_node = helper.make_node(
        "Gemm",
        inputs=[input_name, "W", "B"],
        outputs=[gemm_out],
        alpha=1.0,
        beta=1.0,
    )
    softmax_node = helper.make_node(
        "Softmax",
        inputs=[gemm_out],
        outputs=[output_name],
        axis=1,
    )

    graph = helper.make_graph(
        [gemm_node, softmax_node],
        "sentinel_ci_model",
        [input_tensor],
        [output_tensor],
        initializer=[w_init, b_init],
    )

    # Triton 23.10 ships ONNX Runtime 1.16: IR version <= 9, opset <= 19.
    # Latest onnx pip package defaults to IR 11 / opset 23, so pin both.
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    out_dir = BASE_DIR / "model_repository" / "sentinel_model" / "1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.onnx"
    onnx.save(model, str(out_path))
    print(f"Saved CI model to {out_path}")


if __name__ == "__main__":
    main()
