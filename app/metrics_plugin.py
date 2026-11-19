"""ADK plugin that times every model call and tool call and files it under the current request.

Registered on every Runner (see AgentService). Plugin callbacks run for every
agent, model call and tool the runner manages, sub-agents included. The critic
has its own Runner with the same plugin, so its calls are counted too.
All callbacks return None, so they never change what the agents do.
"""

import time
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from app.config import ModelPrice
from app.errors import classify
from app.metrics import ToolCall, current_metrics, make_llm_call


def _call_key(callback_context: CallbackContext) -> tuple[str, str, str]:
    """Identify one agent's model call: invocation, agent and branch.

    An agent's model calls are sequential, and parallel single-turn sub-agents run
    on separate branches, so this pairs each 'before' with its 'after'.
    """
    ctx = getattr(callback_context, "_invocation_context", None)
    return (callback_context.invocation_id, callback_context.agent_name, str(getattr(ctx, "branch", "")))


def _tool_key(tool: BaseTool, tool_context: ToolContext) -> str:
    """Unique key for an in-flight tool call, safe even when function_call_id is empty."""
    return tool_context.function_call_id or f"{tool.name}:{id(tool_context)}"


class MetricsPlugin(BasePlugin):
    def __init__(self, prices: dict[str, ModelPrice | None]) -> None:
        super().__init__(name="request_metrics")
        self._prices = prices

    # ---------------------------------------------------------------- model calls

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        metrics = current_metrics()
        if metrics is not None:
            metrics.start_model_call(_call_key(callback_context), llm_request.model or "")
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.partial:  # streaming chunk; the final response carries the usage
            return None
        self._record_model_call(callback_context, llm_response.usage_metadata, error=None)
        return None

    async def on_model_error_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ) -> Optional[LlmResponse]:
        self._record_model_call(callback_context, None, error=classify(error))
        return None  # let the error propagate as before

    def _record_model_call(self, callback_context: CallbackContext, usage: Any, error: str | None) -> None:
        metrics = current_metrics()
        if metrics is None:
            return
        start = metrics.pop_model_call(_call_key(callback_context))
        if start is None:
            return
        started, model = start
        metrics.add_llm_call(
            make_llm_call(
                agent=callback_context.agent_name,
                model=model,
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                prices=self._prices,
                error=error,
            )
        )

    # ---------------------------------------------------------------- tool calls

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict]:
        metrics = current_metrics()
        if metrics is not None:
            metrics.start_tool_call(_tool_key(tool, tool_context))
        return None

    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: dict
    ) -> Optional[dict]:
        ok = not (isinstance(result, dict) and result.get("ok") is False)
        self._record_tool_call(tool, tool_context, ok)
        return None

    async def on_tool_error_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, error: Exception
    ) -> Optional[dict]:
        self._record_tool_call(tool, tool_context, ok=False)
        return None

    def _record_tool_call(self, tool: BaseTool, tool_context: ToolContext, ok: bool) -> None:
        metrics = current_metrics()
        if metrics is None:
            return
        started = metrics.pop_tool_call(_tool_key(tool, tool_context))
        if started is None:
            return
        metrics.add_tool_call(
            ToolCall(
                name=tool.name,
                agent=tool_context.agent_name,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                ok=ok,
                is_agent=isinstance(tool, AgentTool),
            )
        )
