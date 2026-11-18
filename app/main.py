"""FastAPI app: GET /health, POST /ask (orchestrator), POST /ask/sql and POST /ask/rag (debugging)."""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from agents.critic import build_critic
from agents.orchestrator import build_orchestrator
from agents.rag_agent import build_rag_agent
from agents.sql_agent import build_sql_agent
from app.agent_service import AgentResult, AgentService
from app.config import get_settings
from app.errors import ApiError, run_agent_safely, set_request_id
from app.orchestration import AgentCritic, Orchestration, build_ask_response
from tools.bigquery_tools import STATE_ATTEMPTS, STATE_LAST_ROWS, STATE_LAST_SQL
from tools.rag_tools import STATE_CHUNKS, STATE_SOURCES
from tools.upstream import STATE_UPSTREAM_ERROR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: load config and build the agents once."""
    s = get_settings()
    s.export_to_environ()
    app.state.request_timeout_seconds = s.request_timeout_seconds
    timeout = s.llm_timeout_seconds
    orchestrator = AgentService(
        build_orchestrator(
            s.orchestrator_model, s.worker_model, s.bq_dataset, s.sql_max_attempts, timeout_seconds=timeout
        )
    )
    critic = AgentCritic(AgentService(build_critic(s.critic_model, timeout_seconds=timeout)))
    app.state.orchestration = Orchestration(orchestrator, critic, s.critic_max_revisions)
    app.state.sql_service = AgentService(
        build_sql_agent(s.worker_model, s.bq_dataset, s.sql_max_attempts, timeout_seconds=timeout)
    )
    app.state.rag_service = AgentService(build_rag_agent(s.worker_model, timeout_seconds=timeout))
    yield


app = FastAPI(title="Agentic Analyst", lifespan=lifespan)


# ---------------------------------------------------------------- request id + errors


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Give every request an id: used in error bodies and logs, returned as X-Request-ID."""
    request.state.request_id = uuid.uuid4().hex
    set_request_id(request.state.request_id)  # for logs in tools, critic and embeddings
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "error_type": exc.error_type, "request_id": exc.request_id},
    )


async def run_agent(request: Request, service: Any, question: str) -> AgentResult:
    """The single path every /ask endpoint uses to call an agent."""
    request_id = request.state.request_id
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
    return result


# ---------------------------------------------------------------- models


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class CriticSummary(BaseModel):
    supported: bool | None  # None = the critic could not run
    unsupported_claims: list[str]
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
    result = await run_agent(request, orchestration, body.question)
    return build_ask_response(result)


@app.post("/ask/sql", response_model=SqlAskResponse)
async def ask_sql(
    body: AskRequest,
    request: Request,
    service: Annotated[AgentService, Depends(get_sql_service)],
) -> SqlAskResponse:
    result = await run_agent(request, service, body.question)
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
    result = await run_agent(request, service, body.question)
    return RagAskResponse(
        answer=result.answer,
        sources=result.state.get(STATE_SOURCES, []),
        chunks=result.state.get(STATE_CHUNKS, []),
    )
