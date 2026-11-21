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
