"""API tests. They use fake services, so no GCP access is needed."""

from fastapi.testclient import TestClient

from app.agent_service import AgentResult
from app.main import app, get_orchestration, get_rag_service, get_sql_service

# Not using `with TestClient(...)`, so the startup hook (which needs .env and GCP) does not run.
client = TestClient(app)
app.state.request_timeout_seconds = 5  # normally set by the startup hook


class FakeService:
    """Stands in for AgentService or Orchestration: both expose ask(question) -> AgentResult."""

    def __init__(self, answer: str, state: dict | None = None) -> None:
        self.answer = answer
        self.state = state or {}
        self.questions: list[str] = []

    async def ask(self, question: str) -> AgentResult:
        self.questions.append(question)
        return AgentResult(answer=self.answer, state=self.state)


def use_fake(answer: str, state: dict | None = None, dependency=get_orchestration) -> FakeService:
    fake = FakeService(answer, state)
    app.dependency_overrides[dependency] = lambda: fake
    return fake


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(response.headers["X-Request-ID"]) == 32


# ---------------------------------------------------------------- /ask (orchestrator)


def test_ask_returns_full_orchestrator_response() -> None:
    state = {
        "route": "both",
        "agents_called": ["rag_agent", "sql_agent"],
        "last_sql": "SELECT 1",
        "last_sql_rows": [{"category": "Underwear", "return_rate": 0.109}],
        "rag_sources": ["returns_policy.md", "shipping_sla.md"],
        "critic_verdict": {"supported": True, "unsupported_claims": [], "error": None},
        "revisions": 0,
    }
    answer = "Underwear is at 10.9% vs a 10.0% limit [returns_policy.md]."
    fake = use_fake(answer, state)

    response = client.post("/ask", json={"question": "  Which categories breach?  "})

    assert response.status_code == 200
    body = response.json()
    metrics = body.pop("metrics")
    assert set(metrics) == {
        "latency_ms", "llm_calls", "input_tokens", "output_tokens", "thinking_tokens",
        "cost_usd", "sql_attempts", "revisions",
    }
    assert metrics["llm_calls"] == 0  # the fake service makes no model calls
    assert body == {
        "answer": answer,
        "route": "both",
        "agents_called": ["rag_agent", "sql_agent"],
        "sql": "SELECT 1",
        "rows": [{"category": "Underwear", "return_rate": 0.109}],
        "sources": ["returns_policy.md", "shipping_sla.md"],
        "cited_sources": ["returns_policy.md"],
        "critic": {"supported": True, "unsupported_claims": [], "revisions": 0},
        "saved_report": None,
    }
    assert fake.questions == ["Which categories breach?"]  # whitespace stripped


def test_ask_out_of_scope_answer() -> None:
    use_fake("I can only answer questions about ShopCo's documents and sales data.", {"route": "none"})

    body = client.post("/ask", json={"question": "What is the weather in Paris?"}).json()

    assert body["route"] == "none"
    assert body["agents_called"] == [] and body["cited_sources"] == []


def test_ask_rejects_empty_question() -> None:
    use_fake("unused")
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_empty_answer_returns_clean_502() -> None:
    use_fake("")
    response = client.post("/ask", json={"question": "Hello?"})
    assert response.status_code == 502
    body = response.json()
    assert body["error_type"] == "empty_answer"
    assert body["request_id"] == response.headers["X-Request-ID"]


# ---------------------------------------------------------------- /ask/sql and /ask/rag (debugging)


def test_ask_sql_returns_answer_sql_rows_and_attempts() -> None:
    state = {
        "sql_attempts": 2,
        "last_sql": "SELECT COUNT(*) AS n FROM `bigquery-public-data.thelook_ecommerce.orders`",
        "last_sql_rows": [{"n": 5}],
    }
    use_fake("There are 5 orders.", state, dependency=get_sql_service)

    response = client.post("/ask/sql", json={"question": "How many orders?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "There are 5 orders.",
        "sql": state["last_sql"],
        "rows": [{"n": 5}],
        "attempts": 2,
    }


def test_ask_sql_when_agent_never_ran_a_query() -> None:
    use_fake("I could not answer that.", {}, dependency=get_sql_service)

    response = client.post("/ask/sql", json={"question": "What is the meaning of life?"})

    assert response.json() == {"answer": "I could not answer that.", "sql": None, "rows": [], "attempts": 0}


def test_ask_rag_returns_answer_sources_and_chunks() -> None:
    chunk = {
        "chunk_id": "returns_policy.md:c3",
        "source": "returns_policy.md",
        "heading": "ShopCo Returns Policy > Maximum acceptable return rate per product category",
        "text": "Jeans: maximum acceptable return rate is 11.0%.",
        "score": 0.0325,
    }
    state = {"rag_sources": ["returns_policy.md"], "rag_chunks": [chunk]}
    use_fake("The maximum is 11.0% [returns_policy.md].", state, dependency=get_rag_service)

    response = client.post("/ask/rag", json={"question": "Max return rate for Jeans?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The maximum is 11.0% [returns_policy.md].",
        "sources": ["returns_policy.md"],
        "chunks": [chunk],
    }


def test_ask_rag_when_nothing_was_retrieved() -> None:
    use_fake("The documents do not contain this information.", {}, dependency=get_rag_service)

    response = client.post("/ask/rag", json={"question": "Who founded ShopCo?"})

    assert response.json() == {
        "answer": "The documents do not contain this information.",
        "sources": [],
        "chunks": [],
    }
