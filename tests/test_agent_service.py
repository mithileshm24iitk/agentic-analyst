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
