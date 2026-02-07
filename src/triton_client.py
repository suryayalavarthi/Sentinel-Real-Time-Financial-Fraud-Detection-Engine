"""
Triton inference client with gRPC primary and HTTP fallback.
Uses lazy loading for tritonclient to avoid import crashes in test environments.
ZERO top-level tritonclient imports.
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np


def _grpc_available() -> bool:
    try:
        import tritonclient.grpc  # noqa: F401
        return True
    except ImportError:
        return False


def _http_available() -> bool:
    try:
        import tritonclient.http  # noqa: F401
        return True
    except ImportError:
        return False


class TritonClient:
    """
    Triton inference client. Uses gRPC for low latency; falls back to HTTP on failure.
    """

    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "sentinel_model",
        model_version: str = "1",
        input_name: str = "float_input",
        output_name: str = "probabilities",
    ) -> None:
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.output_name = output_name
        self._grpc_client: Any = None
        self._http_client: Any = None
        self._use_grpc = True

    def _get_grpc_client(self):
        if not _grpc_available():
            raise RuntimeError("tritonclient.grpc not installed")
        import tritonclient.grpc as grpc_client
        if self._grpc_client is None:
            host, port = self.url.rsplit(":", 1)
            self._grpc_client = grpc_client.InferenceServerClient(url=f"{host}:{port}")
        return self._grpc_client

    def _get_http_client(self):
        if not _http_available():
            raise RuntimeError("tritonclient.http not installed")
        import tritonclient.http as http_client
        if self._http_client is None:
            self._http_client = http_client.InferenceServerClient(url=self.url.replace(":8001", ":8000"))
        return self._http_client

    def predict(self, features: np.ndarray) -> float:
        """
        Run inference on Triton.

        Args:
            features: Shape (1, n_features) or (n_features,) float32 array.

        Returns:
            Fraud probability scalar.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = features.astype(np.float32)

        if self._use_grpc and _grpc_available():
            try:
                return self._predict_grpc(features)
            except Exception:
                self._use_grpc = False

        if _http_available():
            return self._predict_http(features)

        raise RuntimeError("No Triton client available (grpc or http)")

    def _predict_grpc(self, features: np.ndarray) -> float:
        import tritonclient.grpc as grpc_client
        client = self._get_grpc_client()
        inp = grpc_client.InferInput(self.input_name, features.shape, "FP32")
        inp.set_data_from_numpy(features)
        out = grpc_client.InferRequestedOutput(self.output_name)
        result = client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[inp],
            outputs=[out],
        )
        output = result.as_numpy(self.output_name)
        return float(self._extract_probability(output))

    def _predict_http(self, features: np.ndarray) -> float:
        import tritonclient.http as http_client
        client = self._get_http_client()
        inp = http_client.InferInput(self.input_name, features.shape, "FP32")
        inp.set_data_from_numpy(features)
        out = http_client.InferRequestedOutput(self.output_name)
        result = client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[inp],
            outputs=[out],
        )
        output = result.as_numpy(self.output_name)
        return float(self._extract_probability(output))

    def _extract_probability(self, output: np.ndarray) -> float:
        if output.ndim == 2:
            if output.shape[1] == 2:
                return float(output[0, 1])
            return float(output[0, 0])
        return float(output[0])

    def close(self) -> None:
        if self._grpc_client is not None:
            self._grpc_client.close()
            self._grpc_client = None
        self._http_client = None
