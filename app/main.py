"""FastAPI app: GET /health, POST /ask and POST /ask/sql."""

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from agents.assistant import build_assistant
from agents.sql_agent import build_sql_agent
from app.agent_service import AgentService
from app.config import get_settings
from tools.bigquery_tools import STATE_ATTEMPTS, STATE_LAST_ROWS, STATE_LAST_SQL


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: load config and build the agents once."""
    settings = get_settings()
    settings.export_to_environ()
    app.state.agent_service = AgentService(build_assistant(settings.worker_model))
    app.state.sql_service = AgentService(
        build_sql_agent(settings.worker_model, settings.bq_dataset, settings.sql_max_attempts)
    )
    yield


app = FastAPI(title="Agentic Analyst", lifespan=lifespan)


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class AskResponse(BaseModel):
    answer: str


class SqlAskResponse(BaseModel):
    answer: str
    sql: str | None  # last SQL the agent ran (after guardrails), None if it never ran one
    rows: list[dict[str, Any]]
    attempts: int


# Dependencies, so tests can swap in fake agents.
def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_sql_service(request: Request) -> AgentService:
    return request.app.state.sql_service


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> AskResponse:
    result = await service.ask(body.question)
    if not result.answer:
        raise HTTPException(status_code=502, detail="The agent returned no answer.")
    return AskResponse(answer=result.answer)


@app.post("/ask/sql", response_model=SqlAskResponse)
async def ask_sql(
    body: AskRequest,
    service: Annotated[AgentService, Depends(get_sql_service)],
) -> SqlAskResponse:
    result = await service.ask(body.question)
    if not result.answer:
        raise HTTPException(status_code=502, detail="The SQL agent returned no answer.")
    return SqlAskResponse(
        answer=result.answer,
        sql=result.state.get(STATE_LAST_SQL),
        rows=result.state.get(STATE_LAST_ROWS, []),
        attempts=result.state.get(STATE_ATTEMPTS, 0),
    )
