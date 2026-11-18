"""Shared model-call settings for all agents."""

from google.genai import types


def llm_config(timeout_seconds: int | None) -> types.GenerateContentConfig | None:
    """Per-call HTTP timeout for Gemini. google-genai has no timeout by default."""
    if not timeout_seconds:
        return None
    return types.GenerateContentConfig(
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000)  # milliseconds
    )
