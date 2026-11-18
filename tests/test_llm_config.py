"""THINKING_LEVEL and the per-call timeout reach every agent's model requests."""

import asyncio

from agents.critic import build_critic
from agents.llm_config import llm_config
from agents.orchestrator import build_orchestrator
from app.agent_service import AgentService
from tests.scripted_llm import ScriptedLlm


def test_llm_config_none_when_nothing_set():
    assert llm_config(None, None) is None
    assert llm_config(None, "") is None


def test_llm_config_sets_timeout_and_thinking_level():
    cfg = llm_config(90, "LOW")
    assert cfg.http_options.timeout == 90_000
    assert cfg.thinking_config.thinking_level.value == "LOW"


def test_thinking_unset_keeps_model_default():
    cfg = llm_config(90, None)
    assert cfg.thinking_config is None


def test_thinking_level_reaches_orchestrator_sub_agents_and_critic():
    orch_llm = ScriptedLlm(script=[("rag_agent", {"request": "q"}), "done"])
    worker_llm = ScriptedLlm(script=["no tools needed"])
    critic_llm = ScriptedLlm(script=['{"supported": true, "unsupported_claims": []}'])

    orchestrator = build_orchestrator(orch_llm, worker_llm, "p.d", 2, timeout_seconds=90, thinking_level="LOW")
    critic = build_critic(critic_llm, timeout_seconds=90, thinking_level="LOW")
    asyncio.run(AgentService(orchestrator).ask("q"))
    asyncio.run(AgentService(critic).ask("check"))

    for llm in (orch_llm, worker_llm, critic_llm):
        for request in llm.requests:
            assert request.config.thinking_config.thinking_level.value == "LOW"
            assert request.config.http_options.timeout == 90_000
