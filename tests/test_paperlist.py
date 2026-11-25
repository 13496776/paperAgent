import json
import time

import httpx
import pytest

pytest.importorskip("fastapi")
jwt = pytest.importorskip("jwt")
from fastapi.testclient import TestClient

from corecoder import web
from corecoder.llm import LLMResponse, ToolCall
from corecoder.papers.paperlist import search_paperlist
from corecoder.persistent import SessionStore
from corecoder.workflow.engine import ResearchTools, Supervisor


def test_paperlist_mcp_search_and_pagination(monkeypatch):
    monkeypatch.setenv("PAPERLIST_API_KEY", "pl_test")
    calls = []

    def handler(request):
        assert str(request.url) == "https://paperlist.ai/mcp"
        assert request.headers["authorization"] == "Bearer pl_test"
        if request.method == "DELETE":
            calls.append("DELETE")
            return httpx.Response(200)
        message = json.loads(request.content)
        calls.append(message["method"])
        if message["method"] == "notifications/initialized":
            assert request.headers["mcp-session-id"] == "test-session"
            return httpx.Response(202)
        if message["method"] == "initialize":
            return httpx.Response(200, headers={"Mcp-Session-Id": "test-session"}, json={
                "jsonrpc": "2.0", "id": message["id"],
                "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}},
            })
        assert message["params"] == {"name": "search_papers", "arguments": {
            "query": "agents", "fields": "full", "limit": 10, "cursor": "page-2"}}
        result = {"papers": [{"id": "iclr2025-42", "title": "Example", "authors": ["Alice"],
                              "abstract": "Abstract", "links": {"paperlist": "https://paperlist.ai/en/paper/42"}}],
                  "next_cursor": None}
        reply = {"jsonrpc": "2.0", "id": message["id"], "result": {"structuredContent": result}}
        body = ": keepalive\n\nevent: message\ndata: " + json.dumps(reply) + "\n\n"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        data = search_paperlist("agents", cursor="page-2", client=client)
    assert data["papers"][0]["title"] == "Example"
    assert data["papers"][0]["authors"] == ["Alice"]
    assert data["next_cursor"] is None
    assert calls == ["initialize", "notifications/initialized", "tools/call", "DELETE"]


def test_paperlist_missing_key_and_invalid_response(monkeypatch):
    monkeypatch.delenv("PAPERLIST_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PAPERLIST_API_KEY"):
        search_paperlist("agents")
    monkeypatch.setenv("PAPERLIST_API_KEY", "pl_test")
    def handler(request):
        message = json.loads(request.content)
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        result = {"protocolVersion": "2025-06-18"} if message["method"] == "initialize" else {"structuredContent": {}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(RuntimeError, match="论文列表"):
        search_paperlist("agents", client=client)


def test_paperlist_json_text_fallback_and_unsafe_link(monkeypatch):
    monkeypatch.setenv("PAPERLIST_API_KEY", "pl_test")
    def handler(request):
        message = json.loads(request.content)
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26"}
        else:
            data = {"papers": [{"id": "iclr-1", "title": "Test", "url": "javascript:alert(1)"}],
                    "next_cursor": "next"}
            result = {"content": [{"type": "text", "text": json.dumps(data)}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = search_paperlist("test", client=client)
    assert result["papers"][0]["url"] == ""
    assert result["next_cursor"] == "next"


def test_paperlist_web_auth_and_workflow_tool(monkeypatch):
    key = "test-key-that-is-long-enough-for-hs256"
    monkeypatch.setenv("CORECODER_JWT_KEY", key)
    monkeypatch.setenv("CORECODER_JWT_ISSUER", "issuer")
    monkeypatch.setenv("CORECODER_JWT_AUDIENCE", "app")
    monkeypatch.setenv("PAPERLIST_API_KEY", "pl_test")
    monkeypatch.delenv("PAPER_EVALUATION_LOCAL_ONLY", raising=False)
    def search(query, *, cursor=None):
        return {"papers": [{"id": "p1", "title": query}], "next_cursor": cursor}
    monkeypatch.setattr("corecoder.papers.paperlist.search_paperlist", search)
    client = TestClient(web.app)
    assert client.post("/api/corpus/paperlist/search", json={"query": "agents"}).status_code in (401, 403)
    now = int(time.time())
    token = jwt.encode({"sub": "alice", "iss": "issuer", "aud": "app", "iat": now, "exp": now + 60}, key,
                       algorithm="HS256")
    result = client.post("/api/corpus/paperlist/search", json={"query": "agents", "cursor": "next"},
                         headers={"Authorization": f"Bearer {token}"})
    assert result.status_code == 200 and result.json()["next_cursor"] == "next"
    tools = ResearchTools(paperlist_search=search)
    assert tools.execute("retrieval", "search_paperlist", {"query": "agents"})["papers"][0]["id"] == "p1"
    assert "search_online" in [schema["function"]["name"] for schema in web.get_tools("alice").schemas("retrieval")]


def test_paperlist_evidence_reaches_delivery_worker(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")

    class Model:
        def __init__(self):
            self.count = 0

        def chat(self, messages, **kwargs):
            self.count += 1
            if self.count == 1:
                return LLMResponse(content=json.dumps({"tasks": [
                    {"id": "r", "worker": "retrieval", "goal": "search", "depends_on": []},
                    {"id": "d", "worker": "delivery", "goal": "report", "depends_on": ["r"]}]}))
            if self.count == 2:
                return LLMResponse(tool_calls=[ToolCall("call", "search_paperlist", {"query": "agents"})])
            if self.count == 4:
                payload = json.loads(messages[1]["content"])
                assert payload["dependencies"]["r"]["evidence"][0]["result"]["papers"][0]["title"] == "Paper"
            return LLMResponse(content="Done")

    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "session.sqlite")
    tools = ResearchTools(paperlist_search=lambda query, cursor=None: {"papers": [{"title": "Paper"}], "next_cursor": None})
    supervisor = Supervisor(Model(), store, tools)
    supervisor.start("alice", "session", "Search online")
    assert supervisor.resume("alice", "session")["status"] == "completed"
