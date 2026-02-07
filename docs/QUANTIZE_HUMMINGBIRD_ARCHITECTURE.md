# Quantize Pipeline: Hummingbird Re-architecture Plan

## Executive Summary

Re-engineer `quantize.py` to use **Hummingbird** as the primary XGBoost→ONNX conversion engine, resolving the recursion-depth hang. Add a `--lightweight` CLI flag for demo-friendly smaller models, and ensure I/O names align with Triton config (`float_input`, `probabilities`).

---

## 1. Problem & Solution

| Aspect | Current State | Target State |
|--------|---------------|--------------|
| Conversion engine | onnxmltools (recursive `_remap_nodeid` hangs) | Hummingbird (tensor-based, no recursion) |
| Fallback | Hummingbird (plan-B, often uninstalled) | onnxmltools (plan-B if Hummingbird fails) |
| Demo on limited HW | Full model, slow conversion | `--lightweight` → smaller model, fast conversion |
| I/O names | onnxmltools uses `float_input` | Hummingbird may use different names → remap to `float_input` / `probabilities` |

---

## 2. Hummingbird as Primary Engine

### 2.1 Conversion Flow

```
XGBClassifier (or lightweight slice)
    → hummingbird.ml.convert(model, "onnx", extra_config={"onnx_target_opset": 13})
    → HummingbirdONNXModel (or similar)
    → Extract ONNX ModelProto
    → Rename I/O to float_input / probabilities (if needed)
    → Serialize to model.onnx
```

### 2.2 Hummingbird API Notes

- `convert(model, "onnx", extra_config={"onnx_target_opset": 13})` returns a Hummingbird container.
- The container exposes `model` (PyTorch/ONNX graph) or `save(path)`.
- For ONNX backend, Hummingbird typically exports via `torch.onnx.export` or equivalent; input/output names may be generic (e.g. `input`, `output`) or framework-specific.
- **Post-conversion step**: Inspect ONNX graph inputs/outputs; if names ≠ `float_input` / `probabilities`, apply rename via `onnx.helper` or `onnx.compose`.

### 2.3 I/O Name Mapping Strategy

1. After Hummingbird conversion, load the ONNX model and inspect:
   - `model.graph.input[0].name`
   - `model.graph.output[*].name`
2. If input ≠ `float_input` or output ≠ `probabilities`:
   - Use `onnx.helper` to create new `ValueInfoProto` with correct names.
   - Update `model.graph.input` and `model.graph.output`.
   - Rename usages in the graph (e.g. identity nodes or direct references).
3. Alternatively: Use `onnx.compose` or `onnx.utils.extract_model` with input/output name mapping if supported.
4. **Triton contract**: Input `float_input` shape `[batch, 436]`, output `probabilities` shape `[batch, 2]` (binary classifier).

---

## 3. Safety Net: `--lightweight` Mode

### 3.1 Behavior

When `--lightweight` is passed:

1. Load the full XGBoost model from `models/sentinel_fraud_model.pkl`.
2. **Model slicing**: Create a new XGBClassifier with `n_estimators=50` and **retrain** on the same training data (`train_engineered.pkl`) using the same feature pipeline.
3. Convert this smaller model with Hummingbird.
4. Validation (output match, ROC-AUC) uses the **lightweight model** for XGBoost and the exported ONNX for comparison.

### 3.2 Rationale for Retraining vs. Tree Slicing

- `model.set_params(n_estimators=50)` on a fitted model does **not** remove trees; XGBoost stores the full booster.
- Slicing the booster (e.g. keeping first 50 trees from `get_dump()`) would require rebuilding a Booster from parsed dumps—complex and error-prone.
- **Retraining** with `n_estimators=50` yields a smaller, valid XGBoost model that converts quickly and is suitable for demos. Accuracy will be slightly lower but acceptable for Feb 11th demo.

### 3.3 Implementation Sketch

```python
def get_model_for_conversion(model_path: Path, data_path: Path, feature_names: List[str], lightweight: bool) -> XGBClassifier:
    model = joblib.load(model_path)
    if not lightweight:
        return model
    # Retrain smaller model
    df = pd.read_pickle(data_path)
    X = _prepare_features(df, feature_names)
    y = df["isFraud"].values if "isFraud" in df.columns else ...
    small = XGBClassifier(n_estimators=50, max_depth=model.max_depth, ...)  # copy key params
    small.fit(X, y)
    return small
```

---

## 4. Validation: `validate_outputs_match` for Hummingbird

### 4.1 Output Shape Handling

Hummingbird ONNX models may produce:

- **Case A**: Single output, shape `(batch, 2)` (class probabilities) → use as-is.
- **Case B**: Two outputs `(label, probabilities)` → use the probabilities output.
- **Case C**: Single output, shape `(batch,)` (raw scores or label indices) → may need softmax or different extraction.

### 4.2 Robust Extraction Logic

```python
def _extract_fraud_probability(pred_onnx_raw, input_name: str = "float_input") -> np.ndarray:
    """Handle Hummingbird/onnxmltools output shapes."""
    if isinstance(pred_onnx_raw, (list, tuple)):
        # Multiple outputs: (label, probabilities)
        pred = pred_onnx_raw[1] if len(pred_onnx_raw) > 1 else pred_onnx_raw[0]
    else:
        pred = pred_onnx_raw
    if pred.ndim == 1:
        return pred  # regression / single prob
    if pred.shape[1] >= 2:
        return pred[:, 1]  # fraud (positive) probability
    return pred[:, 0]
```

### 4.3 Input Name Discovery

`validate_outputs_match` and `validate_roc_auc` currently hardcode `{"float_input": X}`. For robustness:

1. Inspect ONNX model: `sess.get_inputs()[0].name` to get the actual input name.
2. Use that name when calling `sess.run(None, {input_name: X})`.

This makes validation work regardless of whether the ONNX model was produced by Hummingbird (with possible rename) or onnxmltools.

---

## 5. Fallback: onnxmltools as Plan-B

If Hummingbird fails (e.g. unsupported XGBoost version, conversion error):

1. Catch the exception.
2. Log: "Hummingbird failed, falling back to onnxmltools (may be slow)."
3. Use the existing onnxmltools path with recursion limit increase.
4. Ensure onnxmltools output uses `float_input` / `probabilities` (it already does).

---

## 6. CLI Design

```text
python quantize/quantize.py [--lightweight]
```

- `--lightweight`: Retrain with `n_estimators=50` and convert the smaller model.
- Default: Use full model with Hummingbird primary, onnxmltools fallback.

---

## 7. File Changes Summary

| File | Changes |
|------|---------|
| `quantize/quantize.py` | Hummingbird primary; `--lightweight`; I/O rename; robust `validate_outputs_match`; input-name discovery |
| `requirements.txt` | Add `hummingbird-ml[extra]` (includes xgboost) |
| `model_repository/sentinel_model/config.pbtxt` | No change (keep `float_input`, `probabilities`) |
| `src/triton_client.py` | No change (already expects `float_input`, `probabilities`) |

---

## 8. Execution Order

1. Add `argparse` for `--lightweight`.
2. Implement `get_model_for_conversion(model_path, data_path, feature_names, lightweight)`.
3. Refactor `export_xgboost_to_onnx` to:
   - Try Hummingbird first with `extra_config={"onnx_target_opset": 13}`.
   - Apply I/O rename if needed (float_input, probabilities).
   - Fall back to onnxmltools on Hummingbird failure.
4. Update `validate_outputs_match` and `validate_roc_auc` to discover ONNX input name and handle multiple output shapes.
5. Update `main()` to pass `lightweight` into the conversion and model-loading path.
6. Add `hummingbird-ml[extra]` to `requirements.txt`.

---

## 9. Verification Checklist

- [ ] `python quantize/quantize.py` completes in &lt; 2 min (full model, Hummingbird).
- [ ] `python quantize/quantize.py --lightweight` completes in &lt; 1 min.
- [ ] `model.onnx` has input `float_input`, output `probabilities`.
- [ ] Triton loads and serves inference correctly.
- [ ] `validate_outputs_match` passes for both Hummingbird and onnxmltools outputs.
- [ ] ROC-AUC validation passes (drop &lt; 1%).
