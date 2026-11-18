"""Upstream failures become clean JSON {error, error_type, request_id}, never a bare 500 page."""

import asyncio
import concurrent.futures
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from google.api_core import exceptions as gcp
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors

from app.errors import classify
from app.main import app, get_orchestration, get_rag_service, get_sql_service

client = TestClient(app)


def genai_error(code: int) -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": "boom", "status": "X"}})


@pytest.mark.parametrize(
    "exc, error_type",
    [
        (genai_error(429), "quota_exceeded"),
        (genai_error(401), "auth_error"),
        (genai_error(403), "auth_error"),
        (genai_error(400), "upstream_error"),
        (genai_error(404), "upstream_error"),
        (genai_error(500), "upstream_unavailable"),
        (genai_error(503), "upstream_unavailable"),
        (gcp.TooManyRequests("bq quota"), "quota_exceeded"),
        (gcp.ResourceExhausted("quota"), "quota_exceeded"),
        (gcp.Forbidden("no access"), "auth_error"),
        (gcp.Unauthenticated("who"), "auth_error"),
        (gcp.ServiceUnavailable("down"), "upstream_unavailable"),
        (gcp.DeadlineExceeded("slow"), "timeout"),
        (gcp.BadRequest("bad"), "upstream_error"),
        (auth_exceptions.RefreshError("token expired"), "auth_error"),
        (auth_exceptions.DefaultCredentialsError("no ADC"), "auth_error"),
        (TimeoutError(), "timeout"),
        (asyncio.TimeoutError(), "timeout"),
        (concurrent.futures.TimeoutError(), "timeout"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "upstream_unavailable"),
        (ValueError("bug"), "internal_error"),
    ],
)
def test_classify(exc, error_type):
    assert classify(exc) == error_type


class RaisingService:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def ask(self, question: str):
        raise self.exc


def teardown_function() -> None:
    app.dependency_overrides.clear()


@pytest.mark.parametrize("path, dependency", [
    ("/ask", get_orchestration),
    ("/ask/sql", get_sql_service),
    ("/ask/rag", get_rag_service),
])
@pytest.mark.parametrize(
    "exc, status, error_type",
    [
        (genai_error(429), 503, "quota_exceeded"),
        (auth_exceptions.RefreshError("token expired"), 502, "auth_error"),
        (gcp.DeadlineExceeded("slow"), 503, "timeout"),
        (RuntimeError("unexpected"), 500, "internal_error"),
    ],
)
def test_every_ask_endpoint_returns_clean_json_error(path, dependency, exc, status, error_type):
    app.dependency_overrides[dependency] = lambda: RaisingService(exc)

    response = client.post(path, json={"question": "anything"})

    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error", "error_type", "request_id"}
    assert body["error_type"] == error_type
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "boom" not in body["error"] and "token expired" not in body["error"]  # no internals leaked


def test_failure_is_logged_with_traceback_and_request_id(caplog):
    app.dependency_overrides[get_orchestration] = lambda: RaisingService(genai_error(429))

    with caplog.at_level(logging.ERROR, logger="app.errors"):
        response = client.post("/ask", json={"question": "anything"})

    request_id = response.json()["request_id"]
    record = next(r for r in caplog.records if request_id in r.getMessage())
    assert record.exc_info is not None  # full traceback attached
    assert "quota_exceeded" in record.getMessage()
