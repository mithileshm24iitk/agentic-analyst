"""Tests for picking the final answer out of ADK events."""

from google.adk.events import Event
from google.genai import types

from app.agent_service import final_text


def text_event(text: str, partial: bool = False) -> Event:
    return Event(
        author="assistant",
        partial=partial,
        content=types.Content(role="model", parts=[types.Part(text=text)]),
    )


def test_final_text_returns_final_response() -> None:
    assert final_text([text_event("Hello")]) == "Hello"


def test_final_text_ignores_partial_events() -> None:
    assert final_text([text_event("Hel", partial=True), text_event("Hello")]) == "Hello"


def test_final_text_empty_when_no_events() -> None:
    assert final_text([]) == ""


def authored(author: str, *parts: types.Part) -> Event:
    return Event(author=author, content=types.Content(role="model", parts=list(parts)))


def test_final_text_ignores_sub_agent_final_responses_when_author_given() -> None:
    events = [
        authored("orchestrator", types.Part(text="Draft from orchestrator")),
        authored("sql_agent", types.Part(text="There are 7332 items.")),  # sub-agent, also "final"
    ]
    assert final_text(events, author="orchestrator") == "Draft from orchestrator"
    assert final_text(events) == "There are 7332 items."  # without author: last final event wins


def test_final_text_skips_thought_parts() -> None:
    event = authored("orchestrator", types.Part(text="thinking...", thought=True), types.Part(text="Answer."))
    assert final_text([event], author="orchestrator") == "Answer."


def test_final_text_keeps_previous_answer_when_last_event_is_only_thoughts() -> None:
    events = [
        authored("orchestrator", types.Part(text="Answer.")),
        authored("orchestrator", types.Part(text="still thinking", thought=True)),
    ]
    assert final_text(events, author="orchestrator") == "Answer."
