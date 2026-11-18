"""A fake LLM for running real ADK agents in tests without calling Gemini.

It replays a script: each step is either text (the model answers) or a
(tool_name, args) tuple (the model calls a tool). Agents that share one
ScriptedLlm instance consume the same script in call order.
"""

from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class ScriptedLlm(BaseLlm):
    model: str = "scripted"
    script: list[Any] = []
    requests: list[Any] = []  # every LlmRequest received, for assertions

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        if not self.script:
            raise AssertionError("ScriptedLlm ran out of scripted steps")
        step = self.script.pop(0)
        if isinstance(step, tuple):
            name, args = step
            part = types.Part(function_call=types.FunctionCall(name=name, args=args))
        else:
            part = types.Part(text=step)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))
