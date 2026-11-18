"""retrieve() tool: result shape, session state, never raising. hybrid_search is faked."""

from tests.conftest import make_tool_context
from tools import rag_tools
from tools.rag_tools import STATE_CHUNKS, STATE_QUERIES, STATE_SOURCES, retrieve


def fake_hit(chunk_id: str, source: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source": source,
        "heading": "H",
        "text": f"text of {chunk_id}",
        "score": 0.03,
        "vector_rank": 1,
        "vector_distance": 0.1,
        "bm25_rank": 1,
        "bm25_score": 2.0,
    }


def use_hits(monkeypatch, hits_by_query: dict[str, list[dict]]) -> None:
    monkeypatch.setattr(rag_tools, "hybrid_search", lambda query: hits_by_query[query])


def test_retrieve_returns_chunks_and_sources(monkeypatch):
    use_hits(monkeypatch, {"q": [fake_hit("a.md:c0", "a.md"), fake_hit("b.md:c1", "b.md")]})

    result = retrieve("q", make_tool_context())

    assert result["ok"] is True
    assert result["sources"] == ["a.md", "b.md"]
    assert [c["chunk_id"] for c in result["chunks"]] == ["a.md:c0", "b.md:c1"]
    assert set(result["chunks"][0]) == {"chunk_id", "source", "heading", "text", "score"}


def test_retrieve_accumulates_deduplicated_state_within_one_question(monkeypatch):
    use_hits(
        monkeypatch,
        {
            "first": [fake_hit("a.md:c0", "a.md")],
            "second": [fake_hit("a.md:c0", "a.md"), fake_hit("c.md:c2", "c.md")],
        },
    )
    ctx = make_tool_context("question-1")

    retrieve("first", ctx)
    retrieve("second", ctx)

    assert [c["chunk_id"] for c in ctx.state[STATE_CHUNKS]] == ["a.md:c0", "c.md:c2"]
    assert ctx.state[STATE_SOURCES] == ["a.md", "c.md"]
    assert ctx.state[STATE_QUERIES] == ["first", "second"]


def test_retrieve_state_accumulates_across_invocations_in_one_session(monkeypatch):
    # The orchestrator's revision turn is a new invocation in the same session;
    # evidence from the draft turn must survive it.
    use_hits(monkeypatch, {"draft": [fake_hit("a.md:c0", "a.md")], "revise": [fake_hit("b.md:c0", "b.md")]})
    state: dict = {}

    retrieve("draft", make_tool_context("draft-turn", state))
    retrieve("revise", make_tool_context("revision-turn", state))

    assert state[STATE_SOURCES] == ["a.md", "b.md"]
    assert state[STATE_QUERIES] == ["draft", "revise"]
    assert [c["chunk_id"] for c in state[STATE_CHUNKS]] == ["a.md:c0", "b.md:c0"]


def test_retrieve_returns_error_instead_of_raising(monkeypatch):
    def broken(query):
        raise RuntimeError("index missing")

    monkeypatch.setattr(rag_tools, "hybrid_search", broken)
    ctx = make_tool_context()

    result = retrieve("q", ctx)

    assert result == {"ok": False, "chunks": [], "sources": [], "error": "index missing"}
    assert ctx.state[STATE_QUERIES] == ["q"]


def test_embedding_quota_error_is_fatal_and_recorded(monkeypatch):
    from google.genai import errors as genai_errors

    def quota(query):
        raise genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "X"}})

    monkeypatch.setattr(rag_tools, "hybrid_search", quota)
    ctx = make_tool_context()

    result = retrieve("q", ctx)

    assert result["ok"] is False and "Do not retry" in result["error"]
    assert ctx.state["upstream_error"] == {"error_type": "quota_exceeded", "tool": "retrieve"}
