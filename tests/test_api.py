"""API tests. They use a fake agent, so no GCP access is needed."""

from fastapi.testclient import TestClient

from app.main import app, get_agent_service

# Not using `with TestClient(...)`, so the startup hook (which needs .env and GCP) does not run.
client = TestClient(app)


class FakeAgentService:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions: list[str] = []

    async def ask(self, question: str) -> str:
        self.questions.append(question)
        return self.answer


def use_fake(answer: str) -> FakeAgentService:
    fake = FakeAgentService(answer)
    app.dependency_overrides[get_agent_service] = lambda: fake
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
