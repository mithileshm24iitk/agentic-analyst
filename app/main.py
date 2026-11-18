"""FastAPI app: GET /health and POST /ask."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from agents.assistant import build_assistant
from app.agent_service import AgentService
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: load config and build the agent once."""
    settings = get_settings()
    settings.export_to_environ()
    app.state.agent_service = AgentService(build_assistant(settings.worker_model))
    yield


app = FastAPI(title="Agentic Analyst", lifespan=lifespan)


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class AskResponse(BaseModel):
    answer: str


def get_agent_service(request: Request) -> AgentService:
    """Dependency so tests can swap in a fake agent."""
    return request.app.state.agent_service


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> AskResponse:
    answer = await service.ask(body.question)
    if not answer:
        raise HTTPException(status_code=502, detail="The agent returned no answer.")
    return AskResponse(answer=answer)
