"""Fatal upstream errors inside tools (expired credentials, exhausted quota).

Tools never raise to the agent, but for these errors a retry cannot help, so the
tool tells the agent to stop and records the error in session state. The API
layer (app/main.py run_agent) then returns a clean 502/503 instead of a 200
answer that only says "data unavailable".
"""

import logging
from typing import Any

from app.errors import classify, current_request_id

logger = logging.getLogger(__name__)

STATE_UPSTREAM_ERROR = "upstream_error"
FATAL_ERROR_TYPES = {"auth_error", "quota_exceeded"}


def fatal_error_type(exc: BaseException) -> str | None:
    """'auth_error' or 'quota_exceeded' if the exception is fatal for this request, else None."""
    error_type = classify(exc)
    if error_type in FATAL_ERROR_TYPES:
        logger.error(
            "request_id=%s fatal upstream error (%s) in a tool", current_request_id(), error_type,
            exc_info=exc,
        )
        return error_type
    return None


def fatal_message(error_type: str) -> str:
    what = "Authentication with Google Cloud failed" if error_type == "auth_error" else "A Google Cloud quota was exceeded"
    return f"{what}. Do not retry. Tell the user the data source is unavailable right now."


def record_fatal_error(state: Any, error_type: str, tool_name: str) -> None:
    """Remember the first fatal error of this request in session state."""
    if not state.get(STATE_UPSTREAM_ERROR):
        state[STATE_UPSTREAM_ERROR] = {"error_type": error_type, "tool": tool_name}
