"""Runs an ADK agent and returns its answer plus session state."""

from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "agentic_analyst"
USER_ID = "api-user"


@dataclass
class AgentResult:
    answer: str
    state: dict[str, Any] = field(default_factory=dict)  # final session state, e.g. sql_attempts


def final_text(events: list[Event], author: str | None = None) -> str:
    """Text of the last final-response event, or "" if there is none.

    Sub-agents run inside the same invocation and their last events are also
    "final", so pass `author` (the root agent's name) to ignore them.
    """
    answer = ""
    for event in events:
        if author and event.author != author:
            continue
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(part.text or "" for part in event.content.parts if not part.thought)
            if text:
                answer = text
    return answer


class AgentService:
    """Wraps an ADK Runner with an in-memory session store.

    ask() is the one-shot path: a fresh session per question. The orchestration
    uses the lower-level steps (start_session, run_turn, get_state, update_state)
    to run several turns in one session.
    """

    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent
        self._sessions = InMemorySessionService()
        self._runner = Runner(app_name=APP_NAME, agent=agent, session_service=self._sessions)

    async def start_session(self) -> str:
        session = await self._sessions.create_session(app_name=APP_NAME, user_id=USER_ID)
        return session.id

    async def run_turn(self, session_id: str, message: str) -> str:
        """Send one user message and return the root agent's final text."""
        content = types.Content(role="user", parts=[types.Part(text=message)])
        events = [
            event
            async for event in self._runner.run_async(
                user_id=USER_ID, session_id=session_id, new_message=content
            )
        ]
        return final_text(events, author=self.agent.name)

    async def get_state(self, session_id: str) -> dict[str, Any]:
        session = await self._sessions.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        return dict(session.state) if session else {}

    async def update_state(self, session_id: str, delta: dict[str, Any]) -> None:
        """Write state from outside an agent: append an event carrying a state_delta.

        This is ADK's tracked way to change state; editing session.state directly
        would bypass the session service.
        """
        session = await self._sessions.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        event = Event(author="system", actions=EventActions(state_delta=delta))
        await self._sessions.append_event(session, event)

    async def ask(self, question: str) -> AgentResult:
        session_id = await self.start_session()
        answer = await self.run_turn(session_id, question)
        return AgentResult(answer=answer, state=await self.get_state(session_id))
