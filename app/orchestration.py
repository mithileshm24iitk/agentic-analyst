"""The /ask flow: orchestrator draft -> critic -> at most CRITIC_MAX_REVISIONS revisions.

The revision cap is enforced here in Python, not by the prompt: the loop below can
run at most `max_revisions` extra orchestrator turns, whatever the models say.
"""

import logging
from typing import Any, Awaitable, Callable, Protocol

from agents.critic import CriticVerdict, build_critic_prompt, parse_verdict
from agents.orchestrator import STATE_AGENTS_CALLED
from app.agent_service import AgentResult, AgentService
from app.citations import cited_sources
from tools.bigquery_tools import STATE_LAST_ROWS, STATE_LAST_SQL
from tools.rag_tools import STATE_SOURCES
from tools.report_tools import STATE_SAVED_REPORT

logger = logging.getLogger(__name__)

STATE_ROUTE = "route"
STATE_CRITIC = "critic_verdict"
STATE_REVISIONS = "revisions"

# A critic takes (question, draft answer, session state) and returns a verdict dict:
# {"supported": bool | None, "unsupported_claims": [...], "error": str | None}
Critic = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


class TurnRunner(Protocol):
    """What the orchestration needs from AgentService (tests pass a fake)."""

    async def start_session(self) -> str: ...
    async def run_turn(self, session_id: str, message: str) -> str: ...
    async def get_state(self, session_id: str) -> dict[str, Any]: ...
    async def update_state(self, session_id: str, delta: dict[str, Any]) -> None: ...


def route_from_agents(agents_called: list[str]) -> str:
    """docs / sql / both / none, from what was actually called (not what the model claims)."""
    called = set(agents_called)
    if {"rag_agent", "sql_agent"} <= called:
        return "both"
    if "rag_agent" in called:
        return "docs"
    if "sql_agent" in called:
        return "sql"
    return "none"


def revision_message(verdict: dict[str, Any]) -> str:
    claims = "\n".join(f"- {c}" for c in verdict.get("unsupported_claims", [])) or "- (not specified)"
    return (
        "Reviewer feedback: these claims in your answer are not supported by the evidence "
        f"you gathered:\n{claims}\n\nRewrite your answer. Remove or correct these claims "
        "using only evidence from the specialists. Return the complete revised answer."
    )


class AgentCritic:
    """Runs the critic agent in its own session. Never raises: failures become supported=None."""

    def __init__(self, service: AgentService) -> None:
        self._service = service

    async def __call__(self, question: str, draft: str, state: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._service.ask(build_critic_prompt(question, draft, state))
            verdict: CriticVerdict = parse_verdict(result.answer)
        except Exception as e:
            logger.exception("Critic failed; keeping the draft without a verdict")
            return {"supported": None, "unsupported_claims": [], "error": f"{type(e).__name__}: {e}"}
        return {"supported": verdict.supported, "unsupported_claims": verdict.unsupported_claims, "error": None}


class Orchestration:
    def __init__(self, orchestrator: TurnRunner, critic: Critic, max_revisions: int) -> None:
        self._orchestrator = orchestrator
        self._critic = critic
        self._max_revisions = max_revisions

    async def ask(self, question: str) -> AgentResult:
        orch = self._orchestrator
        session_id = await orch.start_session()

        answer = await orch.run_turn(session_id, question)
        verdict = await self._critic(question, answer, await orch.get_state(session_id))

        revisions = 0
        while verdict["supported"] is False and revisions < self._max_revisions:
            revised = await orch.run_turn(session_id, revision_message(verdict))
            revisions += 1
            if revised:
                answer = revised
            # Re-check so the stored verdict describes the final answer. No further revisions
            # happen once the cap is reached, whatever this verdict says.
            verdict = await self._critic(question, answer, await orch.get_state(session_id))

        state = await orch.get_state(session_id)
        delta = {
            STATE_ROUTE: route_from_agents(state.get(STATE_AGENTS_CALLED, [])),
            STATE_CRITIC: verdict,
            STATE_REVISIONS: revisions,
        }
        await orch.update_state(session_id, delta)
        return AgentResult(answer=answer, state={**state, **delta})


def build_ask_response(result: AgentResult) -> dict[str, Any]:
    """Assemble the /ask JSON body from the final answer and session state."""
    state = result.state
    sources = state.get(STATE_SOURCES, [])
    verdict = state.get(STATE_CRITIC) or {}
    return {
        "answer": result.answer,
        "route": state.get(STATE_ROUTE, "none"),
        "agents_called": state.get(STATE_AGENTS_CALLED, []),
        "sql": state.get(STATE_LAST_SQL),
        "rows": state.get(STATE_LAST_ROWS, []),
        "sources": sources,
        "cited_sources": cited_sources(result.answer, sources),
        "critic": {
            "supported": verdict.get("supported"),
            "unsupported_claims": verdict.get("unsupported_claims", []),
            "revisions": state.get(STATE_REVISIONS, 0),
        },
        "saved_report": state.get(STATE_SAVED_REPORT),
    }
