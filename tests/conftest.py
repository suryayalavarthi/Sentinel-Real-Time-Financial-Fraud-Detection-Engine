"""
Pytest configuration and fixtures.
Mocks Triton client to prevent Connection Refused when running unit tests.
"""
import sys
from unittest.mock import MagicMock

import numpy as np

# Mock tritonclient before any import that might use it (e.g., main.py, src.triton_client)
_mock_grpc = MagicMock()
_mock_grpc_client = MagicMock()
_mock_result = MagicMock()
_mock_result.as_numpy.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
_mock_grpc_client.infer.return_value = _mock_result
_mock_grpc.InferenceServerClient = MagicMock(return_value=_mock_grpc_client)
_mock_grpc.InferInput = MagicMock()
_mock_grpc.InferRequestedOutput = MagicMock()

_mock_http = MagicMock()
_mock_http_client = MagicMock()
_mock_http_client.infer.return_value = _mock_result
_mock_http.InferenceServerClient = MagicMock(return_value=_mock_http_client)
_mock_http.InferInput = MagicMock()
_mock_http.InferRequestedOutput = MagicMock()

sys.modules["tritonclient.grpc"] = _mock_grpc
sys.modules["tritonclient.http"] = _mock_http
