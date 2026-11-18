"""One place that turns upstream failures into clean JSON errors.

Every /ask endpoint runs its agent through run_agent_safely(). Any exception is
logged with the full traceback and the request id, classified, and re-raised as
ApiError, which main.py renders as {error, error_type, request_id}.
"""

import asyncio
import concurrent.futures
import logging
from typing import Awaitable, TypeVar

import httpx
from google.api_core import exceptions as gcp
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors

logger = logging.getLogger("app.errors")

T = TypeVar("T")

# error_type -> (HTTP status, message shown to the client). Details stay in the logs.
ERRORS: dict[str, tuple[int, str]] = {
    "quota_exceeded": (503, "An upstream quota or rate limit was reached. Try again in a minute."),
    "timeout": (503, "An upstream service took too long to respond. Try again."),
    "upstream_unavailable": (503, "An upstream service is temporarily unavailable. Try again."),
    "auth_error": (502, "The server could not authenticate with Google Cloud."),
    "upstream_error": (502, "An upstream service returned an error."),
    "empty_answer": (502, "The agent returned no answer."),
    "internal_error": (500, "Unexpected server error."),
}

_TIMEOUTS = (TimeoutError, asyncio.TimeoutError, concurrent.futures.TimeoutError, httpx.TimeoutException)


class ApiError(Exception):
    def __init__(self, error_type: str, request_id: str) -> None:
        self.error_type = error_type
        self.status_code, self.message = ERRORS[error_type]
        self.request_id = request_id
        super().__init__(self.message)


def classify(exc: BaseException) -> str:
    """Map an exception from Gemini, BigQuery, auth or the network to an error_type."""
    if isinstance(exc, _TIMEOUTS) or isinstance(exc, gcp.DeadlineExceeded):
        return "timeout"
    if isinstance(exc, auth_exceptions.GoogleAuthError):
        return "auth_error"
    if isinstance(exc, genai_errors.APIError):  # Gemini / embeddings (google-genai)
        code = exc.code or 0
        if code == 429:
            return "quota_exceeded"
        if code in (401, 403):
            return "auth_error"
        if code in (500, 502, 503, 504):
            return "upstream_unavailable"
        return "upstream_error"
    if isinstance(exc, (gcp.TooManyRequests, gcp.ResourceExhausted)):  # BigQuery and other GCP APIs
        return "quota_exceeded"
    if isinstance(exc, (gcp.Unauthenticated, gcp.Forbidden, gcp.PermissionDenied)):
        return "auth_error"
    if isinstance(exc, (gcp.ServiceUnavailable, gcp.InternalServerError, gcp.BadGateway)):
        return "upstream_unavailable"
    if isinstance(exc, gcp.GoogleAPICallError):
        return "upstream_error"
    if isinstance(exc, httpx.TransportError):  # connection refused, DNS, reset...
        return "upstream_unavailable"
    return "internal_error"


async def run_agent_safely(call: Awaitable[T], request_id: str) -> T:
    """Await an agent call; on failure log everything and raise a clean ApiError."""
    try:
        return await call
    except Exception as exc:
        error_type = classify(exc)
        logger.exception("request_id=%s agent call failed (%s)", request_id, error_type)
        raise ApiError(error_type, request_id) from exc
