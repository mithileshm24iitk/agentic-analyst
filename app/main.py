"""FastAPI app: GET /health, POST /ask (orchestrator), POST /ask/sql and POST /ask/rag (debugging),
GET /metrics/summary.

Everything except /health needs the X-API-Key header when API_KEY is set
(always in production: the app refuses to start without one).
"""

import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from agents.critic import build_critic
from agents.orchestrator import build_orchestrator
from agents.rag_agent import build_rag_agent
from agents.sql_agent import build_sql_agent
from app.agent_service import AgentResult, AgentService
from app.config import get_settings
from app.errors import ApiError, run_agent_safely, set_request_id
from app.metrics import append_record, finish_request, log_record, response_metrics, start_request
from app.metrics_plugin import MetricsPlugin
from app.metrics_summary import build_summary
from app.orchestration import AgentCritic, Orchestration, build_ask_response
from app.tracing import request_span, setup_tracing
from tools.bigquery_tools import STATE_ATTEMPTS, STATE_LAST_ROWS, STATE_LAST_SQL
from tools.rag_tools import STATE_CHUNKS, STATE_SOURCES
from tools.retrieval.index_check import check_index
from tools.upstream import STATE_UPSTREAM_ERROR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: load config, check the retrieval index, build the agents once.

    Any failure here stops the server before it accepts a request: missing API_KEY in
    production (raised by Settings) or a missing retrieval index (IndexMissing).
    """
    s = get_settings()
    chunks = check_index(s.chroma_dir, s.rag_collection, s.embedding_model, s.embedding_dim)
    logger.info("Retrieval index OK: %d chunks in %s", chunks, s.chroma_dir)
    app.state.api_key = s.api_key.get_secret_value() if s.api_key else None
    if app.state.api_key is None:
        logger.warning("API_KEY is not set: the API is open. Only acceptable for local development.")
    app.state.max_question_chars = s.max_question_chars
    s.export_to_environ()
    setup_tracing(s.tracing_exporter, s.google_cloud_project)
    app.state.request_timeout_seconds = s.request_timeout_seconds
    app.state.metrics_path = s.metrics_path
    llm = {"timeout_seconds": s.llm_timeout_seconds, "thinking_level": s.thinking_level or None}
    plugins = [MetricsPlugin(s.model_prices)]  # one shared plugin: counts every runner's calls

    orchestrator = AgentService(
        build_orchestrator(s.orchestrator_model, s.worker_model, s.bq_dataset, s.sql_max_attempts, **llm),
        plugins,
    )
    critic = AgentCritic(AgentService(build_critic(s.critic_model, **llm), plugins))
    app.state.orchestration = Orchestration(orchestrator, critic, s.critic_max_revisions)
    app.state.sql_service = AgentService(
        build_sql_agent(s.worker_model, s.bq_dataset, s.sql_max_attempts, **llm), plugins
    )
    app.state.rag_service = AgentService(build_rag_agent(s.worker_model, **llm), plugins)
    yield


app = FastAPI(title="Agentic Analyst", lifespan=lifespan)


# ---------------------------------------------------------------- request id + errors


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Give every request an id and enforce X-API-Key on every route except /health."""
    request.state.request_id = uuid.uuid4().hex
    set_request_id(request.state.request_id)  # for logs in tools, critic and embeddings
    if request.url.path != "/health":
        try:
            require_api_key(request, request.headers.get("x-api-key"))
        except ApiError as exc:
            response = await api_error_handler(request, exc)
            response.headers["X-Request-ID"] = request.state.request_id
            return response
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "error_type": exc.error_type, "request_id": exc.request_id},
    )


# ---------------------------------------------------------------- auth


def require_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    """401 unless the X-API-Key header matches API_KEY. No key configured = open (dev only).

    app.state.api_key is set by the startup hook; if it never ran, this raises (500)
    rather than silently allowing the request.
    """
    expected: str | None = request.app.state.api_key
    if expected is None:
        return
    # compare_digest takes the same time wherever the strings differ, so the response
    # time does not reveal how much of a guessed key was right.
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise ApiError("unauthorized", request.state.request_id)


async def run_agent(
    request: Request, service: Any, question: str, endpoint: str, default_route: str | None = None
) -> tuple[AgentResult, dict[str, Any]]:
    """The single path every /ask endpoint uses to call an agent.

    Also opens the request's root trace span and its metrics record, and writes
    one JSON line to METRICS_PATH whether the request succeeds or fails.
    Returns the agent result and the metrics record.
    """
    request_id = request.state.request_id
    limit = request.app.state.max_question_chars
    if len(question) > limit:  # checked before any model call or metrics record
        raise ApiError(
            "question_too_long", request_id, f"The question is too long ({len(question)} characters, max {limit})."
        )
    with request_span(request_id, endpoint) as span:
        start_request(request_id, question, endpoint)
        result: AgentResult | None = None
        status = "cancelled"
        try:
            result = await run_agent_safely(
                service.ask(question), request_id, request.app.state.request_timeout_seconds
            )
            # A tool hit an auth or quota error and told the agent to stop: report that
            # failure instead of a 200 answer that only says the data is unavailable.
            upstream = result.state.get(STATE_UPSTREAM_ERROR)
            if upstream:
                raise ApiError(upstream["error_type"], request_id)
            if not result.answer:
                raise ApiError("empty_answer", request_id)
            status = "ok"
        except ApiError as e:
            status = e.error_type
            raise
        finally:
            record = _close_request(request, span, status, result, default_route)
        return result, record


def _close_request(
    request: Request, span: Any, status: str, result: AgentResult | None, default_route: str | None
) -> dict[str, Any]:
    """Finish the metrics record, tag the root span, append the JSON line."""
    record = finish_request(request.state.request_id, status, result.state if result else {}, default_route)
    span.set_attribute("route", record["route"])
    span.set_attribute("status", status)
    metrics_path = getattr(request.app.state, "metrics_path", None)
    if metrics_path is not None:
        append_record(metrics_path, record)
    log_record(record)  # stdout -> Cloud Logging on Cloud Run
    return record


# ---------------------------------------------------------------- models


class AskRequest(BaseModel):
    # The upper limit is MAX_QUESTION_CHARS, checked in run_agent (it is a setting).
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CriticSummary(BaseModel):
    supported: bool | None  # None = the critic could not run
    unsupported_claims: list[str]
    revisions: int


class RequestMetricsSummary(BaseModel):
    latency_ms: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float | None  # None when a model's price is not in the price table
    sql_attempts: int
    revisions: int


class AskResponse(BaseModel):
    answer: str
    route: str  # docs / sql / both / none
    agents_called: list[str]
    sql: str | None
    rows: list[dict[str, Any]]
    sources: list[str]  # files of every retrieved chunk
    cited_sources: list[str]  # files the answer actually cites (and that were retrieved)
    critic: CriticSummary
    saved_report: str | None
    metrics: RequestMetricsSummary


class SqlAskResponse(BaseModel):
    answer: str
    sql: str | None  # last SQL the agent ran (after guardrails), None if it never ran one
    rows: list[dict[str, Any]]
    attempts: int


class RagChunk(BaseModel):
    chunk_id: str
    source: str
    heading: str
    text: str
    score: float


class RagAskResponse(BaseModel):
    answer: str
    sources: list[str]  # files the retrieved chunks came from
    chunks: list[RagChunk]  # every chunk retrieved for this question


# Dependencies, so tests can swap in fakes.
def get_orchestration(request: Request) -> Orchestration:
    return request.app.state.orchestration


def get_sql_service(request: Request) -> AgentService:
    return request.app.state.sql_service


def get_rag_service(request: Request) -> AgentService:
    return request.app.state.rag_service


# ---------------------------------------------------------------- routes


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    request: Request,
    orchestration: Annotated[Orchestration, Depends(get_orchestration)],
) -> dict[str, Any]:
    result, record = await run_agent(request, orchestration, body.question, "/ask")
    return {**build_ask_response(result), "metrics": response_metrics(record)}


@app.post("/ask/sql", response_model=SqlAskResponse)
async def ask_sql(
    body: AskRequest,
    request: Request,
    service: Annotated[AgentService, Depends(get_sql_service)],
) -> SqlAskResponse:
    result, _ = await run_agent(request, service, body.question, "/ask/sql", default_route="sql")
    return SqlAskResponse(
        answer=result.answer,
        sql=result.state.get(STATE_LAST_SQL),
        rows=result.state.get(STATE_LAST_ROWS, []),
        attempts=result.state.get(STATE_ATTEMPTS, 0),
    )


@app.post("/ask/rag", response_model=RagAskResponse)
async def ask_rag(
    body: AskRequest,
    request: Request,
    service: Annotated[AgentService, Depends(get_rag_service)],
) -> RagAskResponse:
    result, _ = await run_agent(request, service, body.question, "/ask/rag", default_route="docs")
    return RagAskResponse(
        answer=result.answer,
        sources=result.state.get(STATE_SOURCES, []),
        chunks=result.state.get(STATE_CHUNKS, []),
    )


SUMMARY_SCOPE = (
    "Local METRICS_PATH file of the instance that served this request. On Cloud Run each "
    "instance has its own file, lost when it shuts down, so this covers only the current "
    'instance. For all instances, query Cloud Logging: jsonPayload.log_type="request_metrics".'
)


@app.get("/metrics/summary")
def metrics_summary(request: Request) -> dict[str, Any]:
    """Aggregate METRICS_PATH: overall and per route. A missing or empty file gives zero requests."""
    return {"scope": SUMMARY_SCOPE, **build_summary(request.app.state.metrics_path)}
