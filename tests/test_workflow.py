import json

import pytest

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("redis")
pytest.importorskip("jwt")

from corecoder.llm import LLMResponse, ToolCall
from corecoder.persistent import SessionStore
from corecoder.workflow.engine import ResearchTools, Supervisor, validate_plan


class Scripted:
    def __init__(self, results):
        self.results = iter(results)

    def chat(self, **kwargs):
        return next(self.results)


class Retriever:
    def retrieve(self, query):
        return [{"id": "p1", "title": "Paper", "text": query}]


def test_supervisor_plan_approval_dependencies_and_tool_access(tmp_path):
    plan = {"tasks": [{"id": "one", "worker": "retrieval", "goal": "search", "depends_on": []},
                      {"id": "two", "worker": "delivery", "goal": "summarize", "depends_on": ["one"]}]}
    responses = [LLMResponse(content=json.dumps(plan)),
                 LLMResponse(tool_calls=[ToolCall("c", "retrieve_chunks", {"query": "test"})]),
                 LLMResponse(content="Evidence [p1]"), LLMResponse(content="Answer [p1]")]
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(Scripted(responses), store, ResearchTools(Retriever()))
    assert supervisor.start("alice", "session", "question")["status"] == "awaiting_approval"
    assert supervisor.resume("alice", "session")["result"] == "Answer [p1]"
    state = supervisor.get("alice", "session")
    assert state["workflow"]["tasks"][0]["calls"][0]["result"][0]["id"] == "p1"
    assert supervisor.get("bob", "session").get("workflow") is None
    with pytest.raises(ValueError):
        ResearchTools(Retriever()).execute("delivery", "retrieve_chunks", {"query": "x"})


def test_reject_invalid_dependencies():
    with pytest.raises(ValueError):
        validate_plan({"tasks": [{"id": "a", "worker": "retrieval", "goal": "x", "depends_on": ["b"]}]})


def test_pause_during_tool_call_and_resume_without_reexecution(tmp_path):
    import threading

    entered = threading.Event()
    release = threading.Event()
    calls = []

    class BlockingRetriever:
        def retrieve(self, query):
            calls.append(query)
            entered.set()
            assert release.wait(5)
            return [{"id": "paper", "text": "evidence"}]

    plan = {"tasks": [{"id": "one", "worker": "retrieval", "goal": "search", "depends_on": []}]}
    responses = [LLMResponse(content=json.dumps(plan)),
                 LLMResponse(tool_calls=[ToolCall("c", "retrieve_chunks", {"query": "test"})]),
                 LLMResponse(content="Done")]
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(Scripted(responses), store, ResearchTools(BlockingRetriever()))
    supervisor.start("alice", "session", "question")
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(supervisor.resume("alice", "session")))
    thread.start()
    try:
        assert entered.wait(5)
        assert supervisor.pause("alice", "session")["control_requested"] == "pause"
        assert supervisor.get("alice", "session")["workflow"]["control_requested"] == "pause"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert outcome[0]["status"] == "paused"
    result = supervisor.resume("alice", "session")
    assert result["status"] == "completed"
    assert calls == ["test"]
    assert result["tasks"][0]["messages"][-2]["role"] == "tool"


def test_cancel_while_model_call_is_running(tmp_path):
    import threading

    entered = threading.Event()
    release = threading.Event()
    plan = {"tasks": [{"id": "one", "worker": "analysis", "goal": "analyze", "depends_on": []}]}

    class BlockingModel:
        def __init__(self):
            self.count = 0

        def chat(self, **kwargs):
            self.count += 1
            if self.count == 1:
                return LLMResponse(content=json.dumps(plan))
            entered.set()
            assert release.wait(5)
            return LLMResponse(content="finished current request")

    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(BlockingModel(), store)
    supervisor.start("alice", "session", "question")
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(supervisor.resume("alice", "session")))
    thread.start()
    try:
        assert entered.wait(5)
        assert supervisor.cancel("alice", "session")["control_requested"] == "cancel"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert outcome[0]["status"] == "cancelled"


def test_recover_tool_receipt_without_replaying():
    assistant = LLMResponse(tool_calls=[ToolCall("one", "tool", {}), ToolCall("two", "tool", {})]).message
    messages = [assistant]
    task = {"calls": [{"id": "one", "status": "completed", "result": {"answer": 42}}]}
    Supervisor._recover_tool_receipts(task, messages)
    assert json.loads(messages[1]["content"]) == {"answer": 42}
    assert "not executed" in messages[2]["content"]
    Supervisor._recover_tool_receipts(task, messages)
    assert len(messages) == 3


def test_cancel_awaiting_plan_marks_pending_tasks(tmp_path):
    plan = {"tasks": [{"id": "one", "worker": "analysis", "goal": "analyze", "depends_on": []}]}
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(Scripted([LLMResponse(content=json.dumps(plan))]), store)
    supervisor.start("alice", "session", "question")
    result = supervisor.cancel("alice", "session")
    assert result["status"] == "cancelled"
    assert result["tasks"][0]["status"] == "cancelled"


def test_cancel_during_planning(tmp_path):
    import threading

    entered = threading.Event()
    release = threading.Event()

    class BlockingModel:
        def chat(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return LLMResponse(content='{"tasks": []}')

    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(BlockingModel(), store)
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(supervisor.start("alice", "session", "question")))
    thread.start()
    try:
        assert entered.wait(5)
        assert supervisor.cancel("alice", "session")["control_requested"] == "cancel"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert outcome[0]["status"] == "cancelled"


def test_pause_before_tool_execution_does_not_run_it(tmp_path):
    import threading

    entered = threading.Event()
    release = threading.Event()
    tool_calls = []
    plan = {"tasks": [{"id": "one", "worker": "retrieval", "goal": "search", "depends_on": []}]}

    class BlockingModel:
        def __init__(self):
            self.count = 0
            self.last_messages = []

        def chat(self, messages, **kwargs):
            self.count += 1
            if self.count == 1:
                return LLMResponse(content=json.dumps(plan))
            if self.count == 2:
                entered.set()
                assert release.wait(5)
                return LLMResponse(tool_calls=[ToolCall("c", "retrieve_chunks", {"query": "test"})])
            self.last_messages = messages[:]
            return LLMResponse(content="Noted")

    class CountRetriever:
        def retrieve(self, query):
            tool_calls.append(query)
            return []

    model = BlockingModel()
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(model, store, ResearchTools(CountRetriever()))
    supervisor.start("alice", "session", "question")
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(supervisor.resume("alice", "session")))
    thread.start()
    try:
        assert entered.wait(5)
        supervisor.pause("alice", "session")
    finally:
        release.set()
        thread.join(5)
    assert outcome[0]["status"] == "paused"
    assert tool_calls == []
    assert supervisor.resume("alice", "session")["status"] == "completed"
    assert tool_calls == []
    assert any("not executed" in (message.get("content") or "") for message in model.last_messages)


def test_inventory_tool_routes_exact_count_without_embedding(tmp_path):
    class Inventory:
        def list_papers(self):
            return {"count": 3, "papers": [{"id": str(i), "title": f"Paper {i}"} for i in range(3)]}

        def retrieve(self, query):
            raise AssertionError("inventory must not invoke vector search")

    plan = {"tasks": [{"id": "one", "worker": "retrieval", "goal": "统计论文", "depends_on": []},
                      {"id": "two", "worker": "analysis", "goal": "汇报数量", "depends_on": ["one"]}]}
    responses = [LLMResponse(content=json.dumps(plan)), LLMResponse(tool_calls=[ToolCall("c", "list_corpus_papers", {})]),
                 LLMResponse(content="语料库共 3 篇论文"), LLMResponse(content="当前有 3 篇论文")]
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(Scripted(responses), store, ResearchTools(Inventory()))
    supervisor.start("alice", "session", "统计文献库里有几篇论文")
    result = supervisor.resume("alice", "session")
    assert result["status"] == "completed"
    assert result["tasks"][0]["calls"][0]["result"]["count"] == 3
    assert result["result"] == "当前有 3 篇论文"
    assert all(activity["duration_ms"] >= 0 for activity in result["tasks"][0]["activities"])
