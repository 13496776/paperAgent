import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("redis")
jwt = pytest.importorskip("jwt")
fakeredis = pytest.importorskip("fakeredis")
from fastapi.testclient import TestClient

from corecoder import web
from corecoder.llm import LLMResponse
from corecoder.persistent import SessionStore
from corecoder.workflow.engine import Supervisor


class Model:
    def chat(self, **kwargs):
        return LLMResponse(content='{"tasks":[{"id":"a","worker":"delivery","goal":"write","depends_on":[]}]}')


def test_authenticated_session_plan(tmp_path, monkeypatch):
    key = "test-key-that-is-long-enough-for-hs256"
    monkeypatch.setenv("CORECODER_JWT_KEY", key)
    monkeypatch.setenv("CORECODER_JWT_ISSUER", "issuer")
    monkeypatch.setenv("CORECODER_JWT_AUDIENCE", "app")
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "test.sqlite")
    monkeypatch.setattr(web, "get_store", lambda: store)
    monkeypatch.setattr(web, "supervisor", lambda provider, user: Supervisor(Model(), store))
    client = TestClient(web.app)
    assert client.get("/api/sessions/demo").status_code in (401, 403)
    token = jwt.encode({"sub": "alice", "iss": "issuer", "aud": "app", "iat": int(time.time()),
                        "exp": int(time.time()) + 60}, key, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/sessions/demo/plan", json={"question": "Research?"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "awaiting_approval"
    assert client.get("/api/sessions/demo", headers=headers).json()["workflow"]["question"] == "Research?"


def test_local_sign_in_is_one_use_and_produces_valid_jwt(monkeypatch):
    key = "test-key-that-is-long-enough-for-hs256"
    monkeypatch.setenv("CORECODER_JWT_KEY", key)
    monkeypatch.setenv("CORECODER_JWT_ISSUER", "issuer")
    monkeypatch.setenv("CORECODER_JWT_AUDIENCE", "app")
    monkeypatch.setenv("PAPER_LOCAL_BOOTSTRAP_CODE", "test-code")
    monkeypatch.setenv("PAPER_LOCAL_BOOTSTRAP_EXPIRES", str(time.time() + 60))
    client = TestClient(web.app)
    assert client.post("/api/local-session", json={"code": "bad"}).status_code == 401
    assert client.post("/api/local-session", json={"code": "无效"}).status_code == 401
    response = client.post("/api/local-session", json={"code": "test-code"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    claims = jwt.decode(response.json()["token"], key, algorithms=["HS256"], audience="app", issuer="issuer")
    assert claims["sub"] == "local-user"
    assert client.post("/api/local-session", json={"code": "test-code"}).status_code == 401


def test_local_sign_in_is_disabled_or_expired(monkeypatch):
    monkeypatch.delenv("PAPER_LOCAL_BOOTSTRAP_CODE", raising=False)
    monkeypatch.delenv("PAPER_LOCAL_BOOTSTRAP_EXPIRES", raising=False)
    client = TestClient(web.app)
    assert client.post("/api/local-session", json={"code": "test-code"}).status_code == 401
    monkeypatch.setenv("PAPER_LOCAL_BOOTSTRAP_CODE", "test-code")
    monkeypatch.setenv("PAPER_LOCAL_BOOTSTRAP_EXPIRES", str(time.time() - 1))
    assert client.post("/api/local-session", json={"code": "test-code"}).status_code == 401


def test_pause_endpoint_works_during_running_request(tmp_path, monkeypatch):
    import threading

    entered = threading.Event()
    release = threading.Event()
    key = "test-key-that-is-long-enough-for-hs256"
    monkeypatch.setenv("CORECODER_JWT_KEY", key)
    monkeypatch.setenv("CORECODER_JWT_ISSUER", "issuer")
    monkeypatch.setenv("CORECODER_JWT_AUDIENCE", "app")

    class BlockingModel:
        def __init__(self):
            self.count = 0

        def chat(self, **kwargs):
            self.count += 1
            if self.count == 1:
                return LLMResponse(content='{"tasks":[{"id":"a","worker":"analysis","goal":"review","depends_on":[]}]}')
            entered.set()
            assert release.wait(5)
            return LLMResponse(content="Done")

    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "test.sqlite")
    instance = Supervisor(BlockingModel(), store)
    monkeypatch.setattr(web, "get_store", lambda: store)
    monkeypatch.setattr(web, "supervisor", lambda provider, user: instance)
    client = TestClient(web.app)
    now = int(time.time())
    token = jwt.encode({"sub": "alice", "iss": "issuer", "aud": "app", "iat": now, "exp": now + 60}, key, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    plan = client.post("/api/sessions/demo/plan", json={"question": "Research?"}, headers=headers)
    assert plan.status_code == 200
    initial_revision = plan.json()["revision"]
    assert plan.json()["status"] == "awaiting_approval"
    responses = []
    thread = threading.Thread(target=lambda: responses.append(client.post("/api/sessions/demo/resume", json={}, headers=headers)))
    thread.start()
    try:
        assert entered.wait(5)
        running = client.get("/api/sessions/demo", headers=headers).json()["workflow"]
        assert running["status"] == "running"
        assert running["tasks"][0]["status"] == "running"
        assert running["tasks"][0]["activities"][-1]["detail"] == "等待模型决策"
        assert running["revision"] > initial_revision
        pause = client.post("/api/sessions/demo/pause", headers=headers)
        assert pause.status_code == 200
        assert client.get("/api/sessions/demo", headers=headers).json()["workflow"]["control_requested"] == "pause"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert responses[0].json()["status"] == "paused"
    assert responses[0].json()["revision"] > initial_revision


@pytest.mark.parametrize(("deepseek", "openai", "expected"), [
    ("ds-key", "openai-key", "deepseek"),
    ("ds-key", None, "deepseek"),
    (None, "openai-key", "openai"),
    (None, None, "deepseek"),
])
def test_default_provider(monkeypatch, deepseek, openai, expected):
    for name, value in (("DEEPSEEK_API_KEY", deepseek), ("OPENAI_API_KEY", openai)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert TestClient(web.app).get("/api/settings").json()["default_provider"] == expected
    assert web.PlanRequest(question="Research?").provider == expected
    assert web.ResumeRequest().provider == expected
    assert web.PlanRequest(question="Research?", provider="openai").provider == "openai"


def test_web_model_timeout_and_sdk_retries_are_bounded(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("PAPER_LLM_TIMEOUT", raising=False)
    monkeypatch.setattr(web, "get_store", lambda: object())
    monkeypatch.setattr(web, "get_tools", lambda user: object())
    instance = web.supervisor("deepseek", "alice")
    assert instance.llm.extra["timeout"] == 60
    assert instance.llm.client.max_retries == 0
    monkeypatch.setenv("PAPER_LLM_TIMEOUT", "12")
    assert web.supervisor("deepseek", "alice").llm.extra["timeout"] == 12
    monkeypatch.setenv("PAPER_LLM_TIMEOUT", "nan")
    with pytest.raises(ValueError):
        web.supervisor("deepseek", "alice")
