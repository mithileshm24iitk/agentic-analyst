"""Orchestration flow with a fake orchestrator and fake critics. No LLM, no GCP."""

import asyncio
import pytest

from agents.critic import build_critic_prompt, format_evidence, parse_verdict
from app.agent_service import AgentResult
from app.orchestration import (
    Orchestration,
    build_ask_response,
    revision_message,
    route_from_agents,
)


class FakeOrchestrator:
    """Returns scripted answers per turn and holds session state in a dict."""

    def __init__(self, answers: list[str], state: dict | None = None) -> None:
        self.answers = list(answers)
        self.state = state or {}
        self.messages: list[str] = []
        self.sessions = 0

    async def start_session(self) -> str:
        self.sessions += 1
        return "session-1"

    async def run_turn(self, session_id: str, message: str) -> str:
        self.messages.append(message)
        return self.answers.pop(0)

    async def get_state(self, session_id: str) -> dict:
        return dict(self.state)

    async def update_state(self, session_id: str, delta: dict) -> None:
        self.state.update(delta)


class FakeCritic:
    def __init__(self, verdicts: list[dict]) -> None:
        self.verdicts = verdicts
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, question, draft, state):
        self.calls.append((question, draft))
        return self.verdicts[min(len(self.calls), len(self.verdicts)) - 1]


UNSUPPORTED = {"supported": False, "unsupported_claims": ["Jeans max is 15%"], "error": None}
SUPPORTED = {"supported": True, "unsupported_claims": [], "error": None}
NO_VERDICT = {"supported": None, "unsupported_claims": [], "error": "RuntimeError: boom"}


def test_critic_that_always_rejects_causes_exactly_one_revision():
    orch = FakeOrchestrator(["draft", "revised", "should never be used"])
    critic = FakeCritic([UNSUPPORTED])  # always unsupported

    result = asyncio.run(Orchestration(orch, critic, max_revisions=1).ask("q"))

    assert result.answer == "revised"
    assert len(orch.messages) == 2  # question + exactly one revision request
    assert "Jeans max is 15%" in orch.messages[1]
    assert len(critic.calls) == 2  # draft checked, revised answer re-checked
    assert result.state["revisions"] == 1
    assert result.state["critic_verdict"]["supported"] is False  # describes the final answer


def test_cap_is_configurable():
    orch = FakeOrchestrator(["d", "r1", "r2", "r3"])
    result = asyncio.run(Orchestration(orch, FakeCritic([UNSUPPORTED]), max_revisions=2).ask("q"))
    assert result.state["revisions"] == 2
    assert result.answer == "r2"


def test_zero_revisions_allowed_means_no_revision():
    orch = FakeOrchestrator(["draft"])
    result = asyncio.run(Orchestration(orch, FakeCritic([UNSUPPORTED]), max_revisions=0).ask("q"))
    assert result.answer == "draft" and result.state["revisions"] == 0


def test_supported_draft_is_not_revised():
    orch = FakeOrchestrator(["draft"])
    critic = FakeCritic([SUPPORTED])

    result = asyncio.run(Orchestration(orch, critic, max_revisions=1).ask("q"))

    assert result.answer == "draft"
    assert result.state["revisions"] == 0
    assert len(critic.calls) == 1


def test_revision_that_fixes_the_answer_ends_supported():
    orch = FakeOrchestrator(["draft", "fixed"])
    result = asyncio.run(Orchestration(orch, FakeCritic([UNSUPPORTED, SUPPORTED]), max_revisions=1).ask("q"))
    assert result.answer == "fixed"
    assert result.state["critic_verdict"]["supported"] is True


def test_critic_failure_keeps_draft_without_revision():
    orch = FakeOrchestrator(["draft"])
    result = asyncio.run(Orchestration(orch, FakeCritic([NO_VERDICT]), max_revisions=1).ask("q"))
    assert result.answer == "draft"
    assert result.state["revisions"] == 0
    assert result.state["critic_verdict"]["supported"] is None


def test_empty_revision_keeps_previous_answer():
    orch = FakeOrchestrator(["draft", ""])
    result = asyncio.run(Orchestration(orch, FakeCritic([UNSUPPORTED]), max_revisions=1).ask("q"))
    assert result.answer == "draft"


def test_route_and_verdict_are_written_to_session_state():
    orch = FakeOrchestrator(["draft"], state={"agents_called": ["rag_agent", "sql_agent"]})

    asyncio.run(Orchestration(orch, FakeCritic([SUPPORTED]), max_revisions=1).ask("q"))

    assert orch.state["route"] == "both"
    assert orch.state["revisions"] == 0
    assert orch.state["critic_verdict"] == SUPPORTED


@pytest.mark.parametrize(
    "called, route",
    [
        ([], "none"),
        (["rag_agent"], "docs"),
        (["sql_agent", "sql_agent"], "sql"),
        (["rag_agent", "sql_agent"], "both"),
        (["sql_agent", "rag_agent", "rag_agent"], "both"),
    ],
)
def test_route_from_agents(called, route):
    assert route_from_agents(called) == route


def test_revision_message_lists_claims():
    msg = revision_message(UNSUPPORTED)
    assert "- Jeans max is 15%" in msg


# ---------------------------------------------------------------- response assembly


def test_build_ask_response_from_full_state():
    state = {
        "route": "both",
        "agents_called": ["rag_agent", "sql_agent"],
        "last_sql": "SELECT 1",
        "last_sql_rows": [{"n": 1}],
        "rag_sources": ["category_owners.md", "returns_policy.md"],
        "critic_verdict": {"supported": True, "unsupported_claims": [], "error": None},
        "revisions": 1,
        "saved_report": "/tmp/reports/x.md",
    }
    answer = "Underwear breaches [returns_policy.md]; owner Hana [category_owners.md]; bogus [x.md]."

    body = build_ask_response(AgentResult(answer=answer, state=state))

    assert body == {
        "answer": answer,
        "route": "both",
        "agents_called": ["rag_agent", "sql_agent"],
        "sql": "SELECT 1",
        "rows": [{"n": 1}],
        "sources": ["category_owners.md", "returns_policy.md"],
        "cited_sources": ["returns_policy.md", "category_owners.md"],
        "critic": {"supported": True, "unsupported_claims": [], "revisions": 1},
        "saved_report": "/tmp/reports/x.md",
    }


def test_build_ask_response_from_empty_state():
    body = build_ask_response(AgentResult(answer="I can only answer ShopCo questions.", state={}))
    assert body["route"] == "none"
    assert body["agents_called"] == [] and body["rows"] == [] and body["sources"] == []
    assert body["sql"] is None and body["saved_report"] is None
    assert body["critic"] == {"supported": None, "unsupported_claims": [], "revisions": 0}


# ---------------------------------------------------------------- critic helpers


def test_evidence_includes_chunks_and_only_successful_queries():
    state = {
        "rag_chunks": [{"source": "a.md", "chunk_id": "a.md:c0", "text": "Rule A"}],
        "sql_history": [
            {"sql": "SELECT bad", "ok": False, "error_type": "validation", "rows": []},
            {"sql": "SELECT good", "ok": True, "error_type": None, "rows": [{"n": 5}]},
        ],
    }
    evidence = format_evidence(state)
    assert "[a.md] (a.md:c0)\nRule A" in evidence
    assert "SELECT good" in evidence and '{"n": 5}' in evidence
    assert "SELECT bad" not in evidence


def test_evidence_when_nothing_gathered():
    assert format_evidence({}) == "(no evidence was gathered)"
    assert "# Draft answer\nd" in build_critic_prompt("q", "d", {})


def test_parse_verdict_plain_and_fenced():
    assert parse_verdict('{"supported": true, "unsupported_claims": []}').supported is True
    fenced = '```json\n{"supported": false, "unsupported_claims": ["x"]}\n```'
    assert parse_verdict(fenced).unsupported_claims == ["x"]
