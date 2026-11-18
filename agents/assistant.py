"""Phase 1 agent: a single Gemini agent with no tools."""

from google.adk.agents import Agent


def build_assistant(model: str) -> Agent:
    """Create the assistant agent. The model name comes from config."""
    return Agent(
        name="assistant",
        model=model,
        description="Answers business questions in plain language.",
        instruction=(
            "You are a helpful business analytics assistant. "
            "Answer the user's question clearly and concisely. "
            "If you do not know something, say so instead of guessing."
        ),
    )
