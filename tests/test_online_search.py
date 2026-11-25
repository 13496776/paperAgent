import json

import httpx
import pytest

pytest.importorskip("jwt")
pytest.importorskip("redis")

from corecoder.llm import LLMResponse, ToolCall
from corecoder.persistent import SessionStore
from corecoder.workflow.engine import ResearchTools, Supervisor


def test_paperlist_preferred_and_explicit_arxiv():
    calls = []
    def paperlist(query, *, cursor=None):
        calls.append(("paperlist", query, cursor))
        return {"papers": [{"title": "Conference paper"}], "next_cursor": "next", "note": ""}
    def arxiv(query):
        calls.append(("arxiv", query))
        return [{"title": "Preprint"}]
    tools = ResearchTools(online_search=arxiv, paperlist_search=paperlist)
    result = tools.execute("retrieval", "search_online", {"query": "agents"})
    assert result["source"] == "paperlist" and result["next_cursor"] == "next"
    assert calls == [("paperlist", "agents", None)]
    result = tools.execute("retrieval", "search_online", {"query": "agents", "source": "arxiv"})
    assert result["source"] == "arxiv" and calls[-1] == ("arxiv", "agents")
    assert tools.execute("retrieval", "search_online", {"query": "agents", "cursor": "next"})["source"] == "paperlist"


@pytest.mark.parametrize("failure", ["empty", "timeout", "key"])
def test_paperlist_fallback_is_visible(failure):
    calls, activities = [], []
    def paperlist(query, *, cursor=None):
        calls.append("paperlist")
        if failure == "timeout":
            raise httpx.ConnectTimeout("")
        if failure == "key":
            raise RuntimeError("请配置 PAPERLIST_API_KEY")
        return {"papers": [], "next_cursor": None}
    def arxiv(query):
        calls.append("arxiv")
        return [{"id": "2401.12345", "title": "Fallback"}]
    tools = ResearchTools(online_search=arxiv, paperlist_search=paperlist)
    result = tools.execute("retrieval", "search_online", {"query": "agents"},
                           progress=lambda stage, detail: activities.append(detail))
    assert calls == ["paperlist", "arxiv"]
    assert result["source"] == "arxiv" and result["papers"][0]["title"] == "Fallback"
    assert "Paperlist" in result["fallback_reason"]
    assert activities == ["在线搜索：Paperlist", "在线搜索：arXiv（备用）"]


def test_cursor_failure_does_not_mix_sources():
    def fail(query, *, cursor=None):
        raise RuntimeError("unavailable")
    def arxiv(query):
        pytest.fail("Paperlist cursor cannot be used for arXiv")
    tools = ResearchTools(online_search=arxiv, paperlist_search=fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        tools.execute("retrieval", "search_online", {"query": "agents", "cursor": "page-2"})


def test_both_providers_failing_reports_both():
    def paperlist(query, *, cursor=None):
        raise RuntimeError("Paperlist unavailable")
    def arxiv(query):
        raise httpx.ConnectTimeout("")
    tools = ResearchTools(online_search=arxiv, paperlist_search=paperlist)
    with pytest.raises(RuntimeError, match="Paperlist.*arXiv.*ConnectTimeout"):
        tools.execute("retrieval", "search_online", {"query": "agents"})


def test_web_tools_fall_back_without_paperlist_key(monkeypatch):
    pytest.importorskip("fastapi")
    from corecoder import web

    monkeypatch.delenv("PAPERLIST_API_KEY", raising=False)
    monkeypatch.delenv("PAPER_EVALUATION_LOCAL_ONLY", raising=False)
    monkeypatch.setattr("corecoder.papers.ingestion.search_arxiv", lambda query: [{"title": query}])
    result = web.get_tools("alice").execute("retrieval", "search_online", {"query": "agents"})
    assert result["source"] == "arxiv" and result["papers"][0]["title"] == "agents"
    assert "PAPERLIST_API_KEY" in result["fallback_reason"]


def test_online_intent_workflow_and_legacy_tools(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    calls = []
    def paperlist(query, *, cursor=None):
        calls.append("paperlist")
        return {"papers": [{"title": "Evidence"}], "next_cursor": None}
    def arxiv(query, **kwargs):
        calls.append("arxiv")
        return []
    class Model:
        def __init__(self):
            self.step = 0
        def chat(self, messages, **kwargs):
            self.step += 1
            if self.step == 1:
                assert "user's request needs or explicitly permits online search" in messages[0]["content"]
                return LLMResponse(content=json.dumps({"tasks": [
                    {"id": "r", "worker": "retrieval", "goal": "online search", "depends_on": []},
                    {"id": "d", "worker": "delivery", "goal": "answer", "depends_on": ["r"]}]}))
            if self.step == 2:
                assert "search_online" in [schema["function"]["name"] for schema in kwargs["tools"]]
                return LLMResponse(tool_calls=[ToolCall("call", "search_online", {"query": "agents"})])
            if self.step == 4:
                data = json.loads(messages[1]["content"])
                assert data["dependencies"]["r"]["evidence"][0]["result"]["source"] == "paperlist"
            return LLMResponse(content="Done")
    tools = ResearchTools(online_search=arxiv, paperlist_search=paperlist)
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(Model(), store, tools)
    supervisor.start("alice", "session", "请在线搜索 agents 论文")
    result = supervisor.resume("alice", "session")
    assert result["status"] == "completed" and calls == ["paperlist"]
    assert any(activity["detail"] == "在线搜索：Paperlist" for activity in result["tasks"][0]["activities"])
    assert tools.execute("retrieval", "search_arxiv", {"query": "legacy"}) == []
    assert tools.execute("retrieval", "search_paperlist", {"query": "legacy"})["papers"]
