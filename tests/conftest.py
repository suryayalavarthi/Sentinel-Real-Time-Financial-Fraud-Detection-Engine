"""
Pytest configuration and fixtures.
Installs Fake Triton client modules so tests run without real tritonclient connections.
"""
import sys

from tests.fake_triton import make_fake_grpc_module, make_fake_http_module

# Install fake tritonclient modules before any import that might use them.
# This runs at conftest load time, before test collection.
sys.modules["tritonclient"] = type(sys)("tritonclient")
sys.modules["tritonclient.grpc"] = make_fake_grpc_module()
sys.modules["tritonclient.http"] = make_fake_http_module()
