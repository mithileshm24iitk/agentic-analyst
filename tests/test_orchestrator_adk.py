"""Real ADK wiring test: orchestrator -> single-turn sub-agents -> tools -> session state.

The models are ScriptedLlm fakes and BigQuery / search are faked, so no GCP is used,
but the agents, tools, callbacks, Runner and session service are all real ADK.
This proves that state written inside a sub-agent reaches the final session state.
"""

import asyncio
from pathlib import Path

import pytest

from agents.orchestrator import build_orchestrator
from app.agent_service import AgentService
from app.orchestration import Orchestration, build_ask_response
from tests.conftest import DATASET
from tests.scripted_llm import ScriptedLlm
from tools import rag_tools, report_tools

SQL = "SELECT COUNT(*) AS n FROM `bigquery-public-data.thelook_ecommerce.order_items`"
CHUNK = {
    "chunk_id": "returns_policy.md:c4",
    "source": "returns_policy.md",
    "heading": "ShopCo Returns Policy > Maximum acceptable return rate per product category",
    "text": "Underwear: maximum acceptable return rate is 10.0%.",
    "score": 0.03,
    "vector_rank": 1,
    "vector_distance": 0.1,
    "bm25_rank": 1,
    "bm25_score": 3.0,
}


class FixedCritic:
    def __init__(self, *verdicts: dict) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    async def __call__(self, question, draft, state):
        self.calls += 1
        return self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]


SUPPORTED = {"supported": True, "unsupported_claims": [], "error": None}
UNSUPPORTED = {"supported": False, "unsupported_claims": ["made-up number"], "error": None}


@pytest.fixture
def fake_backends(monkeypatch, fake_client, settings, tmp_path) -> Path:
    """Fake BigQuery rows, fake document search, temporary reports folder."""
    fake_client.rows = [{"n": 7332}]
    monkeypatch.setattr(rag_tools, "hybrid_search", lambda query: [CHUNK])
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        report_tools, "get_settings", lambda: settings.model_copy(update={"reports_dir": reports})
    )
    return reports


def run(orchestrator_script: list, worker_script: list, critic) -> tuple[dict, dict]:
    orchestrator = build_orchestrator(
        model=ScriptedLlm(script=orchestrator_script),
        worker_model=ScriptedLlm(script=worker_script),  # shared by both specialists, used in call order
        dataset=DATASET,
        sql_max_attempts=2,
    )
    service = AgentService(orchestrator)
    result = asyncio.run(Orchestration(service, critic, max_revisions=1).ask("Is Underwear over its limit?"))
    return build_ask_response(result), result.state


def test_state_from_both_sub_agents_reaches_final_session_state(fake_backends):
    body, state = run(
        orchestrator_script=[
            ("rag_agent", {"request": "What is the maximum acceptable return rate for Underwear?"}),
            ("sql_agent", {"request": "Count order_items for Underwear"}),
            ("save_report", {"title": "Underwear check", "content": "Underwear is over its limit."}),
            "Underwear's limit is 10.0% [returns_policy.md]; the database shows 7332 items.",
        ],
        worker_script=[
            ("retrieve", {"query": "Underwear maximum acceptable return rate"}),  # rag_agent
            "The limit is 10.0% [returns_policy.md].",
            ("run_query", {"sql": SQL}),  # sql_agent
            "There are 7332 items.",
        ],
        critic=FixedCritic(SUPPORTED),
    )

    # Written by rag_agent's tool, inside the sub-agent:
    assert state["rag_chunks"][0]["chunk_id"] == "returns_policy.md:c4"
    assert state["rag_sources"] == ["returns_policy.md"]
    assert state["rag_queries"] == ["Underwear maximum acceptable return rate"]
    # Written by sql_agent's tool, inside the sub-agent:
    assert state["sql_attempts"] == 1
    assert state["last_sql"].startswith("SELECT COUNT(*) AS n")
    assert state["last_sql_rows"] == [{"n": 7332}]
    assert state["sql_history"][0]["ok"] is True
    # Written by the orchestrator's callback and tool:
    assert state["agents_called"] == ["rag_agent", "sql_agent"]
    assert Path(state["saved_report"]).parent == fake_backends.resolve()
    # Written by the orchestration code after the run:
    assert state["route"] == "both"
    assert state["critic_verdict"] == SUPPORTED
    assert state["revisions"] == 0

    # The answer is the orchestrator's, not a sub-agent's final text.
    assert body["answer"].startswith("Underwear's limit is 10.0%")
    assert body["cited_sources"] == ["returns_policy.md"]
    assert body["rows"] == [{"n": 7332}]


def test_persisted_session_state_matches(fake_backends):
    """Read the state back from the ADK session service itself."""
    orchestrator = build_orchestrator(
        model=ScriptedLlm(script=[("rag_agent", {"request": "limit?"}), "Limit is 10.0% [returns_policy.md]."]),
        worker_model=ScriptedLlm(script=[("retrieve", {"query": "limit"}), "10.0% [returns_policy.md]."]),
        dataset=DATASET,
        sql_max_attempts=2,
    )
    service = AgentService(orchestrator)

    async def flow():
        session_id = await service.start_session()
        await service.run_turn(session_id, "limit?")
        await service.update_state(session_id, {"route": "docs"})
        return await service.get_state(session_id)

    stored = asyncio.run(flow())

    assert stored["rag_sources"] == ["returns_policy.md"]
    assert stored["agents_called"] == ["rag_agent"]
    assert stored["route"] == "docs"  # written via append_event(state_delta)


def test_revision_turn_keeps_evidence_from_draft_turn(fake_backends):
    critic = FixedCritic(UNSUPPORTED, SUPPORTED)
    body, state = run(
        orchestrator_script=[
            ("rag_agent", {"request": "Underwear limit?"}),
            "Underwear's limit is 15% [returns_policy.md].",  # draft with a wrong number
            "Underwear's limit is 10.0% [returns_policy.md].",  # revision turn (new invocation)
        ],
        worker_script=[
            ("retrieve", {"query": "Underwear limit"}),
            "The limit is 10.0% [returns_policy.md].",
        ],
        critic=critic,
    )

    assert critic.calls == 2
    assert state["revisions"] == 1
    assert body["answer"] == "Underwear's limit is 10.0% [returns_policy.md]."
    # Evidence gathered in the draft turn survived the revision turn:
    assert state["rag_sources"] == ["returns_policy.md"]
    assert state["agents_called"] == ["rag_agent"]
    assert body["route"] == "docs"
    assert body["critic"] == {"supported": True, "unsupported_claims": [], "revisions": 1}


def test_out_of_scope_question_calls_no_agents(fake_backends):
    body, state = run(
        orchestrator_script=["I can only answer questions about ShopCo's documents and sales data."],
        worker_script=[],
        critic=FixedCritic(SUPPORTED),
    )
    assert body["route"] == "none"
    assert body["agents_called"] == []
    assert body["sql"] is None and body["sources"] == []
