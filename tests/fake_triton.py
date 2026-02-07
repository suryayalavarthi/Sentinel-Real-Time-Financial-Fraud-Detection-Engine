"""
Fake Triton client for unit tests when tritonclient is missing or unavailable.
Drop-in replacement that avoids real connections.
"""
import types
from typing import Any

import numpy as np


class FakeInferInput:
    """Fake InferInput - accepts shape, ignores set_data_from_numpy."""

    def __init__(self, name: str, shape: tuple, dtype: str) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def set_data_from_numpy(self, data: np.ndarray) -> None:
        pass


class FakeInferRequestedOutput:
    """Fake InferRequestedOutput - accepts output name."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeInferenceResult:
    """Fake inference result with as_numpy returning fake probabilities."""

    def __init__(self) -> None:
        self._probs = np.array([[0.5, 0.5]], dtype=np.float32)

    def as_numpy(self, name: str) -> np.ndarray:
        return self._probs


class FakeInferenceServerClient:
    """Fake InferenceServerClient - infer() returns FakeInferenceResult."""

    def __init__(self, url: str) -> None:
        self.url = url

    def infer(
        self,
        model_name: str,
        model_version: str,
        inputs: list,
        outputs: list,
    ) -> FakeInferenceResult:
        return FakeInferenceResult()

    def close(self) -> None:
        pass


def make_fake_grpc_module() -> types.ModuleType:
    """Create a fake tritonclient.grpc module for sys.modules."""
    mod = types.ModuleType("tritonclient.grpc")
    mod.InferenceServerClient = FakeInferenceServerClient
    mod.InferInput = FakeInferInput
    mod.InferRequestedOutput = FakeInferRequestedOutput
    return mod


def make_fake_http_module() -> types.ModuleType:
    """Create a fake tritonclient.http module for sys.modules."""
    mod = types.ModuleType("tritonclient.http")
    mod.InferenceServerClient = FakeInferenceServerClient
    mod.InferInput = FakeInferInput
    mod.InferRequestedOutput = FakeInferRequestedOutput
    return mod
