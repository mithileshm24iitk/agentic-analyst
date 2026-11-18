"""Runs an ADK agent for one question and returns its answer plus session state."""

from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "agentic_analyst"


@dataclass
class AgentResult:
    answer: str
    state: dict[str, Any] = field(default_factory=dict)  # final session state, e.g. sql_attempts


def final_text(events: list[Event]) -> str:
    """Return the text of the last final-response event, or "" if there is none."""
    answer = ""
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(part.text or "" for part in event.content.parts)
    return answer


class AgentService:
    """Wraps an ADK Runner. Each question gets a fresh in-memory session."""

    def __init__(self, agent: BaseAgent) -> None:
        self._sessions = InMemorySessionService()
        self._runner = Runner(app_name=APP_NAME, agent=agent, session_service=self._sessions)

    async def ask(self, question: str, user_id: str = "api-user") -> AgentResult:
        session = await self._sessions.create_session(app_name=APP_NAME, user_id=user_id)
        message = types.Content(role="user", parts=[types.Part(text=question)])
        events = [
            event
            async for event in self._runner.run_async(
                user_id=user_id, session_id=session.id, new_message=message
            )
        ]
        # Re-read the session: tools write to state during the run.
        finished = await self._sessions.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session.id
        )
        state = dict(finished.state) if finished else {}
        return AgentResult(answer=final_text(events), state=state)
