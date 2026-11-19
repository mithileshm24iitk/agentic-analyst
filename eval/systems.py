"""Build and run the 4 systems compared in Phase 6 (in-process, same code path as /ask).

Systems:
  A: baseline single agent with SQL, RAG and report tools directly attached; no orchestrator or critic.
  B: orchestrator + specialist sub-agents (rag_agent, sql_agent), critic disabled.
  C: full system (orchestrator + specialist sub-agents + critic with CRITIC_MAX_REVISIONS).
  D: full system with ORCHESTRATOR_MODEL set to STRONG_ORCHESTRATOR_MODEL (e.g. gemini-4-argon).
"""

import uuid
from typing import Any

from agents.baseline_agent import build_baseline_agent
from agents.critic import build_critic
from agents.orchestrator import build_orchestrator
from app.agent_service import AgentResult, AgentService
from app.config import Settings, get_settings
from app.errors import ApiError, run_agent_safely, set_request_id
from app.metrics import finish_request, response_metrics, start_request
from app.metrics_plugin import MetricsPlugin
from app.orchestration import STATE_CRITIC, STATE_REVISIONS, STATE_ROUTE, AgentCritic, Orchestration, build_ask_response
from app.tracing import request_span
from tools.bigquery_tools import STATE_HISTORY, STATE_LAST_SQL
from tools.rag_tools import STATE_SOURCES
from tools.upstream import STATE_UPSTREAM_ERROR

SYSTEM_DESCRIPTIONS: dict[str, str] = {
    "A": "Single baseline agent (SQL + RAG tools, no orchestrator, no critic)",
    "B": "Orchestrator + sub-agents, critic disabled",
    "C": "Full system (orchestrator + sub-agents + critic)",
    "D": "Full system with stronger orchestrator model (optional)",
}


async def noop_critic(question: str, draft: str, state: dict[str, Any]) -> dict[str, Any]:
    """Disabled critic for System B: makes no LLM call and never triggers a revision."""
    return {"supported": None, "unsupported_claims": [], "error": None}


def _baseline_route(state: dict[str, Any]) -> str:
    """Derive route ('docs' / 'sql' / 'both' / 'none') from tool state for System A."""
    used_docs = bool(state.get(STATE_SOURCES))
    used_sql = bool(state.get(STATE_LAST_SQL))
    if used_docs and used_sql:
        return "both"
    if used_docs:
        return "docs"
    if used_sql:
        return "sql"
    return "none"


class BaselineWrapper:
    """Adapts System A's single AgentService to produce the same state keys as Orchestration."""

    def __init__(self, service: AgentService) -> None:
        self._service = service

    async def ask(self, question: str) -> AgentResult:
        result = await self._service.ask(question)
        state = dict(result.state)
        state.setdefault(STATE_ROUTE, _baseline_route(state))
        state.setdefault(STATE_CRITIC, {"supported": None, "unsupported_claims": [], "error": None})
        state.setdefault(STATE_REVISIONS, 0)
        return AgentResult(answer=result.answer, state=state)


def build_system(system_id: str, settings: Settings | None = None, plugins: list | None = None) -> Any:
    """Construct system A, B, C or D from config."""
    s = settings or get_settings()
    system_id = system_id.upper()
    if system_id not in SYSTEM_DESCRIPTIONS:
        raise ValueError(f"Unknown system {system_id!r}; expected one of {sorted(SYSTEM_DESCRIPTIONS)}")

    llm = {"timeout_seconds": s.llm_timeout_seconds, "thinking_level": s.thinking_level or None}
    plugins = plugins if plugins is not None else [MetricsPlugin(s.model_prices)]

    if system_id == "A":
        agent = build_baseline_agent(s.worker_model, s.bq_dataset, s.sql_max_attempts, **llm)
        return BaselineWrapper(AgentService(agent, plugins))

    orch_model = s.strong_orchestrator_model if system_id == "D" else (s.orchestrator_model or s.worker_model)
    orchestrator = AgentService(
        build_orchestrator(orch_model, s.worker_model, s.bq_dataset, s.sql_max_attempts, **llm),
        plugins,
    )

    if system_id == "B":
        return Orchestration(orchestrator, noop_critic, max_revisions=0)

    critic = AgentCritic(AgentService(build_critic(s.critic_model or s.worker_model, **llm), plugins))
    return Orchestration(orchestrator, critic, max_revisions=s.critic_max_revisions)


def eval_trace(state: dict[str, Any]) -> dict[str, Any]:
    """The parts of the session state the scorer needs beyond the API response."""
    return {
        "sql_history": list(state.get(STATE_HISTORY, [])),
        "upstream_error": state.get(STATE_UPSTREAM_ERROR),
    }


async def ask_in_process(
    service: Any,
    question: str,
    timeout_seconds: float,
    endpoint: str = "/ask",
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    """Run one question through the exact /ask execution + metrics path (in-process, no HTTP).

    Returns (response_body_or_none, metrics_record, error_type_or_none). The metrics record also
    carries "eval_trace" (every SQL attempt and any recorded upstream error from the session state),
    which the runner moves into the raw result so scoring can look at the whole run.
    """
    request_id = uuid.uuid4().hex
    set_request_id(request_id)
    with request_span(request_id, endpoint) as span:
        start_request(request_id, question, endpoint)
        result: AgentResult | None = None
        try:
            result = await run_agent_safely(service.ask(question), request_id, timeout_seconds)
            upstream = result.state.get(STATE_UPSTREAM_ERROR)
            if upstream:
                raise ApiError(upstream["error_type"], request_id)
            if not result.answer or not result.answer.strip():
                raise ApiError("empty_answer", request_id)
        except ApiError as exc:
            record = finish_request(request_id, exc.error_type, result.state if result else {}, None)
            record["eval_trace"] = eval_trace(result.state if result else {})
            span.set_attribute("route", record["route"])
            span.set_attribute("status", exc.error_type)
            return None, record, exc.error_type

        record = finish_request(request_id, "ok", result.state, None)
        record["eval_trace"] = eval_trace(result.state)
        span.set_attribute("route", record["route"])
        span.set_attribute("status", "ok")
        body = {**build_ask_response(result), "metrics": response_metrics(record)}
        return body, record, None
