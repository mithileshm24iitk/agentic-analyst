"""Shared model-call settings for all agents."""

from google.genai import types


def llm_config(
    timeout_seconds: int | None, thinking_level: str | None = None
) -> types.GenerateContentConfig | None:
    """Per-call settings for Gemini.

    timeout_seconds: HTTP timeout (google-genai has none by default).
    thinking_level: Gemini 3 thinking effort, e.g. "LOW"; None or "" = the model's default.
    """
    if not timeout_seconds and not thinking_level:
        return None
    return types.GenerateContentConfig(
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000) if timeout_seconds else None,  # ms
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None,
    )
