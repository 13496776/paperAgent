import json
import time

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")
jwt = pytest.importorskip("jwt")
fakeredis = pytest.importorskip("fakeredis")
from fastapi.testclient import TestClient

from corecoder import web
from corecoder.llm import LLMResponse, ToolCall
from corecoder.papers.paperlist import get_paperlist
from corecoder.papers.storage import CorpusStore
from corecoder.persistent import SessionStore
from corecoder.workflow.engine import ResearchTools, Supervisor
from tests.test_storage import Encoder


def paper_record(ident="cvpr2026-222437", abstract="Verified abstract text."):
    return {"id": ident, "title": "Verified title", "authors": ["Alice"], "abstract": abstract,
            "url": "https://paperlist.ai/conferences/papers/123", "doi": "", "published_at": None,
            "conference": {"name": "CVPR", "year": 2026}}


def test_paperlist_get_paper_checks_returned_id(monkeypatch):
    monkeypatch.setenv("PAPERLIST_API_KEY", "pl_test")
    def handler(request):
        message = json.loads(request.content)
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        else:
            assert message["params"] == {"name": "get_paper", "arguments": {"id": "cvpr2026-222437"}}
            result = {"structuredContent": {"paper": {"id": "other", "title": "Wrong paper"}}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(RuntimeError, match="ID"):
        get_paperlist("cvpr2026-222437", client=client)


def test_import_saves_abstract_with_scope_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "papers.sqlite"))
    monkeypatch.setattr("corecoder.papers.paperlist.get_paperlist", lambda ident: paper_record(ident))
    encoder = Encoder()
    monkeypatch.setattr(web, "get_models", lambda: (encoder, None))
    first = web.import_paperlist("alice", "cvpr2026-222437")
    assert first["created"] and first["content_scope"] == "abstract" and first["status"] == "ready"
    assert first["page_count"] is None and first["external_id"] == "cvpr2026-222437"
    assert first["year"] == 2026 and first["date_source"] == "paperlist_conference_year"
    store = CorpusStore(tmp_path / "papers.sqlite")
    chunks = store.chunks("alice", first["id"])
    assert chunks[0]["text"] == "Verified abstract text." and chunks[0]["content_scope"] == "abstract"
    assert store.vector_search("alice", [1, 0], 5)
    assert store.papers("bob") == []
    assert not web.import_paperlist("alice", "cvpr2026-222437")["created"]
    assert len(store.papers("alice")) == 1


def test_metadata_only_does_not_index_and_can_upgrade_later(tmp_path, monkeypatch):
    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "papers.sqlite"))
    record = paper_record(abstract="")
    monkeypatch.setattr("corecoder.papers.paperlist.get_paperlist", lambda ident: record)
    monkeypatch.setattr(web, "get_models", lambda: pytest.fail("metadata-only import should not load models"))
    first = web.import_paperlist("alice", record["id"])
    assert first["content_scope"] == first["status"] == "metadata_only" and first["chunks"] == 0
    store = CorpusStore(tmp_path / "papers.sqlite")
    store.ensure_indexed("alice", None)
    assert store.papers("alice")[0]["status"] == "metadata_only"
    record["abstract"] = "Newly available abstract."
    monkeypatch.setattr(web, "get_models", lambda: (Encoder(), None))
    upgraded = web.import_paperlist("alice", record["id"])
    assert not upgraded["created"] and upgraded["id"] == first["id"]
    assert upgraded["status"] == "ready" and upgraded["content_scope"] == "abstract"
    assert store.chunks("alice", first["id"])[0]["text"] == "Newly available abstract."
    assert len(store.papers("alice")) == 1


def test_paperlist_import_recovers_after_index_failure(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path / "papers.sqlite")
    encoder = Encoder()
    monkeypatch.setattr(web, "corpus_store", lambda: store)
    monkeypatch.setattr(web, "get_models", lambda: (encoder, None))
    monkeypatch.setattr("corecoder.papers.paperlist.get_paperlist", lambda ident: paper_record(ident))
    original = store._collection
    def fail(name, path=None):
        raise RuntimeError("index unavailable")
    monkeypatch.setattr(store, "_collection", fail)
    with pytest.raises(RuntimeError, match="index unavailable"):
        web.import_paperlist("alice", "cvpr2026-222437")
    assert store.papers("alice")[0]["status"] == "indexing"
    monkeypatch.setattr(store, "_collection", original)
    retried = web.import_paperlist("alice", "cvpr2026-222437")
    assert retried["status"] == "ready" and not retried["created"]
    assert len(store.papers("alice")) == 1


def test_model_load_failure_keeps_metadata_and_allows_retry(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path / "papers.sqlite")
    monkeypatch.setattr(web, "corpus_store", lambda: store)
    monkeypatch.setattr("corecoder.papers.paperlist.get_paperlist", lambda ident: paper_record(ident))
    def fail():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(web, "get_models", fail)
    with pytest.raises(RuntimeError, match="题录已保存.*model unavailable"):
        web.import_paperlist("alice", "cvpr2026-222437")
    assert store.papers("alice")[0]["status"] == "metadata_only"
    monkeypatch.setattr(web, "get_models", lambda: (Encoder(), None))
    recovered = web.import_paperlist("alice", "cvpr2026-222437")
    assert recovered["status"] == "ready" and recovered["content_scope"] == "abstract"
    assert not recovered["created"] and len(store.papers("alice")) == 1


def test_search_then_import_rejects_unverified_id_and_preserves_session_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLIST_API_KEY", "pl_test")
    monkeypatch.delenv("PAPER_EVALUATION_LOCAL_ONLY", raising=False)
    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "papers.sqlite"))
    monkeypatch.setattr("corecoder.papers.paperlist.get_paperlist", lambda ident: pytest.fail("cached results must not be fetched again"))
    monkeypatch.setattr(web, "get_models", lambda: (Encoder(), None))
    known = "cvpr2026-222437"
    tools = ResearchTools(online_search=lambda query: [],
                          paperlist_search=lambda query, cursor=None: {"papers": [paper_record(known)], "next_cursor": None},
                          paperlist_import=lambda ident, record=None: web.import_paperlist("alice", ident, record=record))
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "papers.sqlite")

    class Model:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content=json.dumps({"tasks": [
                    {"id": "search", "worker": "retrieval", "goal": "search", "depends_on": []},
                    {"id": "save", "worker": "ingestion", "goal": "save", "depends_on": ["search"]},
                    {"id": "report", "worker": "delivery", "goal": "report", "depends_on": ["save"]}]}))
            if self.calls == 2:
                return LLMResponse(tool_calls=[ToolCall("search-call", "search_online", {"query": "agents"})])
            if self.calls == 4:
                return LLMResponse(tool_calls=[ToolCall("good", "import_paperlist", {"id": known}),
                                               ToolCall("bad", "import_paperlist", {"id": "invented-99"})])
            if self.calls == 6:
                receipts = json.loads(messages[1]["content"])["dependencies"]["save"]["evidence"]
                assert [receipt["status"] for receipt in receipts] == ["completed", "failed"]
                assert receipts[0]["result"]["content_scope"] == "abstract"
            return LLMResponse(content="Done")

    supervisor = Supervisor(Model(), store, tools)
    supervisor.start("alice", "session", "在线搜索并保存这些论文")
    result = supervisor.resume("alice", "session")
    saved, rejected = result["tasks"][1]["calls"]
    assert saved["status"] == "completed" and saved["result"]["content_scope"] == "abstract"
    assert rejected["status"] == "failed" and "未出现在本会话" in rejected["error"]
    assert len(CorpusStore(tmp_path / "papers.sqlite").papers("alice")) == 1
    supervisor._validate_paperlist_import("alice", "session", {"question": "保存之前的结果", "tasks": []}, known)
    with pytest.raises(ValueError, match="未出现在本会话"):
        supervisor._validate_paperlist_import("bob", "session", {"question": "保存之前的结果", "tasks": []}, known)

    class FollowupModel:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content=json.dumps({"tasks": [
                    {"id": "history", "worker": "retrieval", "goal": "find previous results", "depends_on": []},
                    {"id": "save", "worker": "ingestion", "goal": "save", "depends_on": ["history"]}]}))
            if self.calls == 2:
                return LLMResponse(tool_calls=[ToolCall("history", "list_paperlist_search_results", {})])
            if self.calls == 4:
                evidence = json.loads(messages[1]["content"])["dependencies"]["history"]["evidence"]
                assert evidence[0]["result"]["papers"][0]["id"] == known
                return LLMResponse(tool_calls=[ToolCall("save", "import_paperlist", {"id": known})])
            return LLMResponse(content="Done")

    followup = Supervisor(FollowupModel(), store, tools)
    followup.start("alice", "session", "把刚才的论文入库")
    completed = followup.resume("alice", "session")
    assert completed["tasks"][1]["calls"][0]["result"]["created"] is False
    monkeypatch.delenv("PAPERLIST_API_KEY")
    cached = web.get_tools("alice").execute("ingestion", "import_paperlist", {"id": known},
                                              verified_paper=paper_record(known))
    assert cached["created"] is False and cached["content_scope"] == "abstract"


def test_previous_results_do_not_skip_a_newer_empty_search(tmp_path):
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "session.sqlite")
    def workflow(ident, papers):
        return {"id": ident, "revision": 1, "tasks": [{"calls": [
            {"status": "completed", "name": "search_online", "result": {"source": "paperlist", "papers": papers}}]}]}
    store.save_completed_workflow("alice", "session", workflow("old", [{"id": "old", "title": "Old"}]))
    store.save_completed_workflow("alice", "session", workflow("new", []))
    supervisor = Supervisor(None, store)
    assert supervisor._previous_paperlist_results("alice", "session", {}) == {"papers": [], "total": 0, "next_offset": None}
    with pytest.raises(ValueError, match="未出现在本会话"):
        supervisor._validate_paperlist_import("alice", "session", {"question": "保存刚才的论文", "tasks": []}, "old")


def test_previous_results_paginate_and_deduplicate(tmp_path):
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "session.sqlite")
    papers = [{"id": f"cvpr-{i}", "title": f"Paper {i}"} for i in range(60)]
    workflow = {"id": "search", "revision": 1, "tasks": [{"calls": [
        {"status": "completed", "name": "search_paperlist", "result": {"papers": papers + papers[:1]}}]}]}
    store.save_completed_workflow("alice", "session", workflow)
    supervisor = Supervisor(None, store)
    first = supervisor._previous_paperlist_results("alice", "session", {"limit": 50})
    second = supervisor._previous_paperlist_results("alice", "session", {"offset": first["next_offset"], "limit": 50})
    assert (first["total"], len(first["papers"]), first["next_offset"]) == (60, 50, 50)
    assert (len(second["papers"]), second["next_offset"]) == (10, None)
    assert supervisor._previous_paperlist_results("bob", "session", {})["papers"] == []
    with pytest.raises(ValueError):
        supervisor._previous_paperlist_results("alice", "session", {"offset": True})


def test_paperlist_import_endpoint_requires_auth(tmp_path, monkeypatch):
    key = "test-key-that-is-long-enough-for-hs256"
    monkeypatch.setenv("CORECODER_JWT_KEY", key)
    monkeypatch.setenv("CORECODER_JWT_ISSUER", "issuer")
    monkeypatch.setenv("CORECODER_JWT_AUDIENCE", "app")
    monkeypatch.setattr(web, "import_paperlist", lambda user, ident: {"owner": user, "external_id": ident})
    client = TestClient(web.app)
    assert client.post("/api/corpus/paperlist/import", json={"id": "cvpr2026-222437"}).status_code in (401, 403)
    now = int(time.time())
    token = jwt.encode({"sub": "alice", "iss": "issuer", "aud": "app", "iat": now, "exp": now + 60}, key,
                       algorithm="HS256")
    response = client.post("/api/corpus/paperlist/import", json={"id": "cvpr2026-222437"},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200 and response.json()["owner"] == "alice"
