"""API key auth, question length limit and production-mode startup refusal. No GCP access."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as main
from app.agent_service import AgentResult
from app.config import Settings
from app.main import app, get_orchestration, get_rag_service, get_sql_service

KEY = "k" * 40
client = TestClient(app)
app.state.request_timeout_seconds = 5  # normally set by the startup hook


class FakeService:
    async def ask(self, question: str) -> AgentResult:
        return AgentResult(answer="An answer.", state={})


@pytest.fixture(autouse=True)
def protected_app(monkeypatch, tmp_path):
    """The app as in production: a key is configured, agents are fakes."""
    app.state.api_key = KEY
    app.state.metrics_path = tmp_path / "requests.jsonl"
    for dependency in (get_orchestration, get_sql_service, get_rag_service):
        app.dependency_overrides[dependency] = FakeService
    yield
    app.dependency_overrides.clear()


PROTECTED = [
    ("post", "/ask"),
    ("post", "/ask/sql"),
    ("post", "/ask/rag"),
    ("get", "/metrics/summary"),
    ("get", "/docs"),
    ("get", "/redoc"),
    ("get", "/openapi.json"),
]


def call(method: str, path: str, headers: dict | None = None, question: str = "How many orders?"):
    if method == "post":
        return client.post(path, json={"question": question}, headers=headers or {})
    return client.get(path, headers=headers or {})


# ---------------------------------------------------------------- auth


@pytest.mark.parametrize("method, path", PROTECTED)
def test_missing_key_is_401(method, path):
    response = call(method, path)

    assert response.status_code == 401
    body = response.json()
    assert body["error_type"] == "unauthorized"
    assert body["request_id"] == response.headers["X-Request-ID"]


@pytest.mark.parametrize("method, path", PROTECTED)
@pytest.mark.parametrize("wrong", ["wrong", KEY[:-1], KEY + "x", KEY.upper(), "", "ключ".encode()])  # raw non-ASCII bytes: 401, not 500
def test_wrong_key_is_401(method, path, wrong):
    assert call(method, path, {"X-API-Key": wrong}).status_code == 401


@pytest.mark.parametrize("method, path", PROTECTED)
def test_right_key_is_allowed(method, path):
    assert call(method, path, {"X-API-Key": KEY}).status_code == 200


def test_health_stays_open():
    response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_auth_is_checked_before_the_body():
    # An unauthenticated caller learns nothing about validation rules.
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 401


def test_key_is_compared_with_compare_digest(monkeypatch):
    seen = []
    real = main.secrets.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(main.secrets, "compare_digest", spy)
    call("post", "/ask", {"X-API-Key": KEY})

    assert seen == [(KEY.encode(), KEY.encode())]


def test_no_key_configured_means_open_in_development(monkeypatch):
    app.state.api_key = None
    assert call("post", "/ask").status_code == 200


def test_unauthorized_request_is_not_metered(tmp_path):
    call("post", "/ask")
    assert not (tmp_path / "requests.jsonl").exists()


# ---------------------------------------------------------------- question length


def test_question_at_the_limit_is_accepted():
    assert call("post", "/ask", {"X-API-Key": KEY}, question="q" * 2000).status_code == 200


@pytest.mark.parametrize("path", ["/ask", "/ask/sql", "/ask/rag"])
def test_question_over_the_limit_is_422(path):
    response = call("post", path, {"X-API-Key": KEY}, question="q" * 2001)

    assert response.status_code == 422
    body = response.json()
    assert body["error_type"] == "question_too_long"
    assert "2001" in body["error"] and "2000" in body["error"]


def test_limit_comes_from_settings(monkeypatch):
    app.state.max_question_chars = 10
    assert call("post", "/ask", {"X-API-Key": KEY}, question="q" * 10).status_code == 200
    assert call("post", "/ask", {"X-API-Key": KEY}, question="q" * 11).status_code == 422


def test_limit_counts_the_question_after_stripping_whitespace(monkeypatch):
    app.state.max_question_chars = 10
    assert call("post", "/ask", {"X-API-Key": KEY}, question="   " + "q" * 10 + "   ").status_code == 200


def test_too_long_question_is_not_metered(tmp_path):
    call("post", "/ask", {"X-API-Key": KEY}, question="q" * 2001)
    assert not (tmp_path / "requests.jsonl").exists()


# ---------------------------------------------------------------- settings and startup


def make_settings(**overrides) -> Settings:
    values = dict(
        _env_file=None,
        google_cloud_project="test-project",
        google_cloud_location="global",
        worker_model="test-model",
        embedding_model="fake-embedding",
        embedding_location="global",
    )
    return Settings(**{**values, **overrides})


def test_development_without_key_is_allowed():
    s = make_settings(app_env="development")
    assert s.api_key is None and s.max_question_chars == 2000


@pytest.mark.parametrize("key", [None, "", "   "])
def test_production_without_key_refuses(key):
    with pytest.raises(ValidationError, match="APP_ENV=production requires API_KEY"):
        make_settings(app_env="production", api_key=key)


def test_production_with_short_key_refuses():
    with pytest.raises(ValidationError, match="at least 32 characters"):
        make_settings(app_env="production", api_key="short")


def test_key_is_trimmed_of_surrounding_whitespace():
    s = make_settings(app_env="production", api_key=KEY + "\n")
    assert s.api_key.get_secret_value() == KEY


def test_rejected_key_is_not_printed_in_the_error():
    secret = "not-long-enough-but-secret"
    with pytest.raises(ValidationError) as info:
        make_settings(app_env="production", api_key=secret)
    assert secret not in str(info.value)


def test_production_with_key_is_allowed():
    s = make_settings(app_env="production", api_key=KEY)
    assert s.api_key.get_secret_value() == KEY
    assert KEY not in repr(s)  # SecretStr keeps it out of logs and tracebacks


def test_settings_read_app_env_and_key_from_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValidationError, match="requires API_KEY"):
        make_settings()


def test_app_refuses_to_start_in_production_without_key(monkeypatch):
    # The startup hook builds Settings first, so the server never accepts a request.
    monkeypatch.setattr(main, "get_settings", lambda: make_settings(app_env="production"))
    with pytest.raises(ValidationError, match="requires API_KEY"):
        with TestClient(app):
            pass
