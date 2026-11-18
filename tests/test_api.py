"""API tests. They use fake agents, so no GCP access is needed."""

from fastapi.testclient import TestClient

from app.agent_service import AgentResult
from app.main import app, get_agent_service, get_sql_service

# Not using `with TestClient(...)`, so the startup hook (which needs .env and GCP) does not run.
client = TestClient(app)


class FakeAgentService:
    def __init__(self, answer: str, state: dict | None = None) -> None:
        self.answer = answer
        self.state = state or {}
        self.questions: list[str] = []

    async def ask(self, question: str) -> AgentResult:
        self.questions.append(question)
        return AgentResult(answer=self.answer, state=self.state)


def use_fake(answer: str, state: dict | None = None, dependency=get_agent_service) -> FakeAgentService:
    fake = FakeAgentService(answer, state)
    app.dependency_overrides[dependency] = lambda: fake
    return fake


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_agent_answer() -> None:
    fake = use_fake("42")
    response = client.post("/ask", json={"question": "  What is the answer?  "})
    assert response.status_code == 200
    assert response.json() == {"answer": "42"}
    assert fake.questions == ["What is the answer?"]  # whitespace stripped


def test_ask_rejects_empty_question() -> None:
    use_fake("unused")
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_returns_502_when_agent_gives_no_answer() -> None:
    use_fake("")
    response = client.post("/ask", json={"question": "Hello?"})
    assert response.status_code == 502


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

    assert response.json() == {
        "answer": "I could not answer that.",
        "sql": None,
        "rows": [],
        "attempts": 0,
    }
