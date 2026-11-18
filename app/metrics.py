"""Per-request metrics: every LLM call and tool call, gathered under the request id.

How calls are matched to a request: main.py starts a RequestMetrics for the
request id (a ContextVar set by the middleware). The ADK plugin in
metrics_plugin.py looks up the current request id on every model / tool callback.
The orchestrator, its sub-agents, the critic (its own Runner and session) and
tools in ADK's thread pool all run inside the request's context, so they all
land in the same RequestMetrics.
"""

import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ModelPrice
from app.costs import call_cost_usd, total_cost_usd
from app.errors import current_request_id


@dataclass
class LlmCall:
    agent: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cached_tokens: int
    tokens_per_second: float | None  # output tokens / latency
    cost_usd: float | None
    error: str | None = None
    traffic_type: str | None = None  # e.g. ON_DEMAND; shows which Vertex capacity served the call


@dataclass
class ToolCall:
    name: str
    agent: str
    duration_ms: float
    ok: bool
    is_agent: bool  # a sub-agent called as a tool (its time includes its own LLM calls)


@dataclass
class RequestMetrics:
    request_id: str
    question: str
    endpoint: str
    started: float = field(default_factory=time.perf_counter)
    llm_calls: list[LlmCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_llm_call(self, call: LlmCall) -> None:
        with self._lock:  # tools run in a thread pool; keep appends safe
            self.llm_calls.append(call)

    def add_tool_call(self, call: ToolCall) -> None:
        with self._lock:
            self.tool_calls.append(call)


_active: dict[str, RequestMetrics] = {}
_active_lock = threading.Lock()


def start_request(request_id: str, question: str, endpoint: str) -> RequestMetrics:
    metrics = RequestMetrics(request_id=request_id, question=question, endpoint=endpoint)
    with _active_lock:
        _active[request_id] = metrics
    return metrics


def current_metrics() -> RequestMetrics | None:
    """The metrics of the request this code is running for, if any."""
    with _active_lock:
        return _active.get(current_request_id())


def make_llm_call(
    agent: str,
    model: str,
    latency_ms: float,
    usage: Any,
    prices: dict[str, ModelPrice | None],
    error: str | None = None,
) -> LlmCall:
    """Build an LlmCall from a Gemini usage_metadata object (or None)."""
    input_tokens = (getattr(usage, "prompt_token_count", None) or 0) if usage else 0
    output_tokens = (getattr(usage, "candidates_token_count", None) or 0) if usage else 0
    thinking_tokens = (getattr(usage, "thoughts_token_count", None) or 0) if usage else 0
    cached_tokens = (getattr(usage, "cached_content_token_count", None) or 0) if usage else 0
    traffic = getattr(usage, "traffic_type", None) if usage else None
    seconds = latency_ms / 1000
    return LlmCall(
        agent=agent,
        model=model,
        latency_ms=round(latency_ms, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        cached_tokens=cached_tokens,
        tokens_per_second=round(output_tokens / seconds, 1) if seconds > 0 else None,
        cost_usd=call_cost_usd(model, input_tokens, output_tokens, thinking_tokens, cached_tokens, prices),
        error=error,
        traffic_type=getattr(traffic, "value", traffic),
    )


def finish_request(
    request_id: str,
    status: str,
    state: dict[str, Any],
    default_route: str | None,
) -> dict[str, Any]:
    """Close the request's metrics and return the record to log (one JSON line)."""
    with _active_lock:
        metrics = _active.pop(request_id, None)
    if metrics is None:  # should not happen; keep the API working anyway
        metrics = RequestMetrics(request_id=request_id, question="", endpoint="")

    from app.orchestration import route_from_agents  # local import: avoids a cycle

    llm, tools = metrics.llm_calls, metrics.tool_calls
    agents_called = state.get("agents_called") or [t.name for t in tools if t.is_agent]
    route = state.get("route") or default_route or route_from_agents(agents_called)
    verdict = state.get("critic_verdict") or {}

    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": metrics.endpoint,
        "question": metrics.question,
        "route": route,
        "agents_called": agents_called,
        "status": status,
        "latency_ms": round((time.perf_counter() - metrics.started) * 1000, 1),
        "llm_calls": len(llm),
        "input_tokens": sum(c.input_tokens for c in llm),
        "output_tokens": sum(c.output_tokens for c in llm),
        "thinking_tokens": sum(c.thinking_tokens for c in llm),
        "cost_usd": total_cost_usd([c.cost_usd for c in llm]),
        "llm_time_ms": round(sum(c.latency_ms for c in llm), 1),
        "tool_time_ms": round(sum(t.duration_ms for t in tools if not t.is_agent), 1),
        "sql_attempts": state.get("sql_attempts", 0),
        "revisions": state.get("revisions", 0),
        "critic_supported": verdict.get("supported"),
        "critic_unsupported_claims": len(verdict.get("unsupported_claims", [])),
        "llm_detail": [asdict(c) for c in llm],
        "tool_detail": [asdict(t) for t in tools],
    }


def response_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """The short metrics object returned in the /ask response."""
    return {
        "latency_ms": record["latency_ms"],
        "llm_calls": record["llm_calls"],
        "input_tokens": record["input_tokens"],
        "output_tokens": record["output_tokens"],
        "thinking_tokens": record["thinking_tokens"],
        "cost_usd": record["cost_usd"],
        "sql_attempts": record["sql_attempts"],
        "revisions": record["revisions"],
    }


_write_lock = threading.Lock()


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line. Never raises: losing a metrics line must not fail a request."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _write_lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        logging.getLogger(__name__).exception("request_id=%s could not write metrics", record.get("request_id"))


LOG_TYPE = "request_metrics"  # filter in Cloud Logging: jsonPayload.log_type="request_metrics"


def log_record(record: dict[str, Any], stream=None) -> None:
    """Print the record as one JSON line on stdout. Never raises.

    On Cloud Run every stdout line that is a single JSON object becomes a structured
    log entry (jsonPayload) in Cloud Logging; "severity" and "message" are special
    fields. Unlike the JSONL file, these lines survive the instance and cover all instances.
    """
    entry = {
        **record,
        "severity": "INFO" if record.get("status") == "ok" else "WARNING",
        "message": (
            f"{LOG_TYPE} {record.get('endpoint')} status={record.get('status')} "
            f"route={record.get('route')} latency_ms={record.get('latency_ms')}"
        ),
        "log_type": LOG_TYPE,
    }
    try:
        line = json.dumps(entry, default=str, ensure_ascii=False)
        out = stream or sys.stdout
        with _write_lock:  # one whole line per request, never interleaved
            out.write(line + "\n")
            out.flush()
    except Exception:
        logging.getLogger(__name__).exception("request_id=%s could not log metrics", record.get("request_id"))
