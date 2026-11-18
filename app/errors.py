"""One place that turns upstream failures into clean JSON errors.

Every /ask endpoint runs its agent through run_agent_safely(). Any exception is
logged with the full traceback and the request id, classified, and re-raised as
ApiError, which main.py renders as {error, error_type, request_id}. Fatal errors
that tools caught and recorded in state (auth, quota) are raised the same way.
"""

import asyncio
import concurrent.futures
import logging
from contextvars import ContextVar
from typing import Awaitable, TypeVar

import aiohttp
import httpx
from google.api_core import exceptions as gcp
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors

logger = logging.getLogger("app.errors")

T = TypeVar("T")

# The current request's id, set by the middleware in main.py. Context variables follow
# the request into ADK's tool thread pool (it copies the context), so tool and
# embedding logs can include it too.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def current_request_id() -> str:
    return _request_id.get()

# error_type -> (HTTP status, message shown to the client). Details stay in the logs.
ERRORS: dict[str, tuple[int, str]] = {
    "unauthorized": (401, "Missing or invalid API key. Send it in the X-API-Key header."),
    "question_too_long": (422, "The question is too long."),
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
    def __init__(self, error_type: str, request_id: str, message: str | None = None) -> None:
        self.error_type = error_type
        self.status_code, default_message = ERRORS[error_type]
        self.message = message or default_message  # only ever our own text, never exception details
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
        if code in (408, 504):  # e.g. DEADLINE_EXCEEDED: our per-call timeout, enforced by Vertex AI
            return "timeout"
        if code in (500, 502, 503):
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
    # Connection refused, DNS, reset... (google-genai uses aiohttp; other clients use httpx)
    if isinstance(exc, (httpx.TransportError, aiohttp.ClientConnectionError)):
        return "upstream_unavailable"
    return "internal_error"


async def run_agent_safely(call: Awaitable[T], request_id: str, timeout_seconds: float) -> T:
    """Await an agent call with a deadline; on failure log everything and raise a clean ApiError.

    The deadline keeps a slow request from running past the platform's own timeout
    (Cloud Run: 300s by default), where it would be killed without a JSON body.
    """
    try:
        return await asyncio.wait_for(call, timeout=timeout_seconds)
    except Exception as exc:
        error_type = classify(exc)
        logger.exception("request_id=%s agent call failed (%s)", request_id, error_type)
        raise ApiError(error_type, request_id) from exc
