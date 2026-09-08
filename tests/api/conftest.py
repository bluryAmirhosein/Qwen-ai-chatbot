import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.router import api_router


@pytest.fixture
def app():
    """A fresh FastAPI app per test, wired the same way app.main is, but
    without importing app.main itself (which also configures logging and
    imports uvicorn — irrelevant for endpoint-level tests).
    """
    test_app = FastAPI()
    test_app.include_router(api_router, prefix="/api/v1")
    yield test_app
    test_app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)
