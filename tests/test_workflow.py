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
    report_path = next((tmp_path / "workflow_results").glob("*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["workflow"]["tasks"][0]["calls"][0]["arguments"] == {"query": "test"}
    assert report["workflow"]["tasks"][0]["calls"][0]["result"][0]["text"] == "test"
    assert supervisor.get("bob", "session").get("workflow") is None
    with pytest.raises(ValueError):
        ResearchTools(Retriever()).execute("delivery", "retrieve_chunks", {"query": "x"})


def test_reject_invalid_dependencies():
    with pytest.raises(ValueError):
        validate_plan({"tasks": [{"id": "a", "worker": "retrieval", "goal": "x", "depends_on": ["b"]}]})


def test_reorder_forward_dependencies_and_reject_cycles():
    tasks = validate_plan({"tasks": [{"id": "b", "worker": "delivery", "goal": "answer", "depends_on": ["a"]},
                                     {"id": "a", "worker": "retrieval", "goal": "search", "depends_on": []}]})
    assert [t["id"] for t in tasks] == ["a", "b"]
    with pytest.raises(ValueError, match="cycle"):
        validate_plan({"tasks": [{"id": "a", "worker": "retrieval", "goal": "x", "depends_on": ["b"]},
                                 {"id": "b", "worker": "delivery", "goal": "y", "depends_on": ["a"]}]})


def test_plan_repair_and_worker_receives_conversation(tmp_path):
    plan = {"tasks": [{"id": "one", "worker": "analysis", "goal": "answer", "depends_on": []}]}
    class Model:
        def __init__(self):
            self.count = 0
            self.worker_context = None
        def chat(self, messages, **kwargs):
            self.count += 1
            if self.count == 1:
                return LLMResponse(content='{"tasks": []}')
            if 'research supervisor' in messages[0]['content']:
                if self.count == 2:
                    assert "Fix the plan" in messages[-1]["content"]
                return LLMResponse(content=json.dumps(plan))
            self.worker_context = json.loads(messages[1]["content"])
            return LLMResponse(content="answer")
    model = Model()
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(model, store)
    supervisor.start('alice', 'session', '只使用本地语料')
    supervisor.resume('alice', 'session')
    supervisor.start('alice', 'session', '这篇论文呢？')
    supervisor.resume('alice', 'session')
    assert model.worker_context['recent_history'][0]['content'] == '只使用本地语料'


def test_compressed_summary_reaches_worker(tmp_path):
    plan = {"tasks": [{"id": "one", "worker": "analysis", "goal": "answer", "depends_on": []}]}
    class Model:
        def __init__(self):
            self.contexts = []
        def chat(self, messages, **kwargs):
            system = messages[0]['content']
            if 'research supervisor' in system:
                return LLMResponse(content=json.dumps(plan))
            if 'Summarize the research conversation' in system:
                return LLMResponse(content='只使用本地论文；研究 DGM')
            self.contexts.append(json.loads(messages[1]['content']))
            return LLMResponse(content='回答')
    model = Model()
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / 'memory.sqlite')
    supervisor = Supervisor(model, store)
    for i in range(8):
        supervisor.start('alice', 'session', f'问题 {i}')
        supervisor.resume('alice', 'session')
    state = supervisor.get('alice', 'session')
    assert state['summary_id'] and len(state['messages']) <= 10
    assert model.contexts[-1]['conversation_summary'] == '只使用本地论文；研究 DGM'


def test_evidence_reaches_delivery_through_analysis(tmp_path):
    plan = {"tasks": [{"id": "r", "worker": "retrieval", "goal": "find", "depends_on": []},
                      {"id": "a", "worker": "analysis", "goal": "compare", "depends_on": ["r"]},
                      {"id": "d", "worker": "delivery", "goal": "answer", "depends_on": ["a"]}]}
    class Model:
        def chat(self, messages, **kwargs):
            system = messages[0]['content']
            if 'research supervisor' in system:
                return LLMResponse(content=json.dumps(plan))
            if 'retrieval research worker' in system and messages[-1]['role'] != 'tool':
                return LLMResponse(tool_calls=[ToolCall('c', 'retrieve_chunks', {'query': 'x'})])
            if 'delivery research worker' in system:
                assert json.loads(messages[1]['content'])['previous_evidence'][0]['id'] == 'p1'
            return LLMResponse(content='answer')
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / 'memory.sqlite')
    supervisor = Supervisor(Model(), store, ResearchTools(Retriever()))
    supervisor.start('alice', 'session', '比较')
    assert supervisor.resume('alice', 'session')['status'] == 'completed'


def test_tool_budgets_are_workflow_wide_and_survive_resume(tmp_path):
    plan = {"tasks": [{"id": "one", "worker": "retrieval", "goal": "search", "depends_on": []}]}
    calls = [ToolCall(str(i), "retrieve_chunks", {"query": "test"}) for i in range(51)]
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(Scripted([LLMResponse(content=json.dumps(plan)), LLMResponse(tool_calls=calls),
                                    LLMResponse(content="partial evidence")]),
                            store, ResearchTools(Retriever()))
    supervisor.start("alice", "session", "question")
    result = supervisor.resume("alice", "session")
    workflow = supervisor.get("alice", "session")["workflow"]
    assert result["status"] == "completed" and result["budget_exhausted"]
    assert "不完整" in result["result"] and len(workflow["tasks"][0]["calls"]) == 50
    assert len(supervisor.get("alice", "session")["workflow"]["tasks"][0]["calls"]) == 50


def test_total_tool_budget_and_final_model_decision(tmp_path):
    class Tools:
        def schemas(self, worker):
            return [{'type': 'function', 'function': {'name': f'tool{i}', 'parameters': {'type': 'object'}}}
                    for i in range(2)]
        def execute(self, worker, name, args, progress=None):
            return "ok"

    plan = {"tasks": [{"id": "one", "worker": "retrieval", "goal": "search", "depends_on": []}]}
    calls = [ToolCall(str(i), f"tool{i // 50}", {}) for i in range(100)]
    for extra in (False, True):
        store = SessionStore(fakeredis.FakeRedis(), tmp_path / f"memory-{extra}.sqlite")
        final = LLMResponse(tool_calls=[ToolCall("51", "tool5", {})]) if extra else LLMResponse(content="done")
        supervisor = Supervisor(Scripted([LLMResponse(content=json.dumps(plan)), LLMResponse(tool_calls=calls), final]),
                                store, Tools())
        supervisor.start("alice", "session", "question")
        if extra:
            with pytest.raises(RuntimeError, match="预算用尽"):
                supervisor.resume("alice", "session")
        else:
            assert supervisor.resume("alice", "session")["result"].endswith("done")
        assert len(supervisor.get("alice", "session")["workflow"]["tasks"][0]["calls"]) == 100


def test_failed_tool_calls_count_across_tasks_and_show_error(tmp_path):
    class FailingRetriever:
        def retrieve(self, query):
            raise TimeoutError()

    plan = {"tasks": [{"id": str(i), "worker": "retrieval", "goal": "search", "depends_on": []} for i in range(2)]}
    first = [ToolCall(str(i), "retrieve_chunks", {"query": "test"}) for i in range(50)]
    responses = [LLMResponse(content=json.dumps(plan)), LLMResponse(tool_calls=first), LLMResponse(content="failed"),
                 LLMResponse(content="无法取得证据")]
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(Scripted(responses), store, ResearchTools(FailingRetriever()))
    supervisor.start("alice", "session", "question")
    result = supervisor.resume("alice", "session")
    workflow = supervisor.get("alice", "session")["workflow"]
    assert workflow["tasks"][0]["calls"][0]["error"] == "TimeoutError"
    assert workflow["tasks"][1]["status"] == "completed" and result["budget_exhausted"]


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


def test_long_workflow_renews_session_lock(tmp_path, monkeypatch):
    import threading
    import time

    from corecoder.workflow import engine

    entered = threading.Event()
    release = threading.Event()

    class SlowModel:
        def chat(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return LLMResponse(content='{"tasks":[{"id":"one","worker":"analysis","goal":"answer","depends_on":[]}]}')

    monkeypatch.setattr(engine, "SESSION_LOCK_SECONDS", 1)
    monkeypatch.setattr(engine, "SESSION_LOCK_RENEW_SECONDS", 0.1)
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    supervisor = Supervisor(SlowModel(), store)
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(supervisor.start("alice", "session", "question")))
    thread.start()
    try:
        assert entered.wait(2)
        time.sleep(1.2)
        with pytest.raises(RuntimeError, match="already running"):
            supervisor.start("alice", "session", "another question")
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert outcome[0]["status"] == "awaiting_approval"


def test_retrieval_timeout_records_failure_and_blocks_overlapping_retry(tmp_path, monkeypatch):
    import threading
    import time

    entered = threading.Event()
    release = threading.Event()
    calls = []

    class SlowRetriever:
        user = "alice"

        def retrieve_with_progress(self, query, progress):
            calls.append(query)
            progress("slow", "慢速检索")
            entered.set()
            assert release.wait(5)
            progress("late", "超时后的进度")
            return [{"id": "late", "text": "late"}]

    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("PAPER_RETRIEVAL_TIMEOUT", "0.05")
    plan = {"tasks": [{"id": "one", "worker": "retrieval", "goal": "search", "depends_on": []}]}
    responses = [LLMResponse(content=json.dumps(plan)),
                 LLMResponse(tool_calls=[ToolCall("c", "retrieve_chunks", {"query": "test"})]),
                 LLMResponse(content="检索超时，证据不足")]
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    tools = ResearchTools(SlowRetriever())
    supervisor = Supervisor(Scripted(responses), store, tools)
    supervisor.start("alice", "session", "question")
    try:
        result = supervisor.resume("alice", "session")
        assert entered.is_set()
        call = result["tasks"][0]["calls"][0]
        assert call["status"] == "failed" and "超过" in call["error"]
        assert any(activity["stage"] == "slow" for activity in result["tasks"][0]["activities"])
        with pytest.raises(RuntimeError, match="后台运行"):
            ResearchTools(SlowRetriever()).execute("retrieval", "retrieve_chunks", {"query": "retry"})
    finally:
        release.set()
    deadline = time.monotonic() + 2
    while threading.active_count() and time.monotonic() < deadline:
        try:
            assert tools.execute("retrieval", "retrieve_chunks", {"query": "after"}, progress=lambda *_: None) == [{"id": "late", "text": "late"}]
            break
        except RuntimeError as exc:
            if "后台运行" not in str(exc):
                raise
            time.sleep(0.01)
    else:
        pytest.fail("background retrieval did not finish")
    assert calls == ["test", "after"]
    saved = supervisor.get("alice", "session")["workflow"]
    assert not any(activity["stage"] == "late" for activity in saved["tasks"][0]["activities"])


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


def test_chinese_instruction_is_applied_to_planning_and_resumed_tasks(tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            assert "请用中文回答。" in messages[0]["content"]
            if self.calls == 1:
                return LLMResponse(content='{"tasks":[{"id":"a","worker":"analysis","goal":"analyze","depends_on":[]}]}')
            assert messages[0]["content"].count("请用中文回答。") == 1
            return LLMResponse(content="结果")

    redis = fakeredis.FakeRedis()
    store = SessionStore(redis, tmp_path / "memory.sqlite")
    supervisor = Supervisor(Model(), store)
    supervisor.start("alice", "session", "question")
    key = store._key("alice", "session")
    state = json.loads(redis.get(key))
    state["workflow"]["tasks"][0]["messages"] = [
        {"role": "system", "content": "Legacy worker instructions."}, {"role": "user", "content": "question"}]
    redis.set(key, json.dumps(state), ex=3600)
    assert supervisor.resume("alice", "session")["result"] == "结果"
    saved = supervisor.get("alice", "session")["workflow"]["tasks"][0]["messages"]
    assert saved[0]["content"].count("请用中文回答。") == 1


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


def test_plan_schema_repairs_invalid_dependency_before_persist(tmp_path):
    bad = {'tasks': [{'id': 'a', 'worker': 'analysis', 'goal': 'answer', 'depends_on': 'none'}]}
    good = {'tasks': [{'id': 'a', 'worker': 'analysis', 'goal': 'answer', 'depends_on': []}]}
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / 'db.sqlite')
    supervisor = Supervisor(Scripted([LLMResponse(content=json.dumps(bad)), LLMResponse(content=json.dumps(good))]), store)
    workflow = supervisor.start('u', 's', 'question')
    assert workflow['tasks'][0]['depends_on'] == []
    assert workflow['status'] == 'awaiting_approval'


def test_invalid_tool_arguments_repaired_before_execution(tmp_path):
    calls = []
    class CountRetriever:
        def retrieve(self, query):
            calls.append(query)
            return [{'id': 'v', 'text': 'valid'}]
    plan = {'tasks': [{'id': 'r', 'worker': 'retrieval', 'goal': 'find', 'depends_on': []}]}
    replies = [LLMResponse(content=json.dumps(plan)),
               LLMResponse(tool_calls=[ToolCall('bad', 'retrieve_chunks', {'query': 42})]),
               LLMResponse(tool_calls=[ToolCall('good', 'retrieve_chunks', {'query': 'right'})]),
               LLMResponse(content='done')]
    supervisor = Supervisor(Scripted(replies), SessionStore(fakeredis.FakeRedis(), tmp_path / 'db.sqlite'), ResearchTools(CountRetriever()))
    supervisor.start('u', 's', 'q')
    workflow = supervisor.resume('u', 's')
    assert calls == ['right']
    assert [call['id'] for call in workflow['tasks'][0]['calls']] == ['good']
    assert not any(msg.get('tool_calls', [{}])[0].get('id') == 'bad' for msg in workflow['tasks'][0]['messages'] if msg.get('tool_calls'))


def test_invalid_retrieval_result_retried_before_state(tmp_path, monkeypatch):
    monkeypatch.setattr('corecoder.workflow.engine.time.sleep', lambda _: None)
    class Retriever:
        def __init__(self): self.calls = 0
        def retrieve(self, query):
            self.calls += 1
            return {'invalid': True} if self.calls == 1 else [{'id': 'v', 'text': 'valid'}]
    retriever = Retriever()
    plan = {'tasks': [{'id': 'r', 'worker': 'retrieval', 'goal': 'find', 'depends_on': []}]}
    responses = [LLMResponse(content=json.dumps(plan)), LLMResponse(tool_calls=[ToolCall('c', 'retrieve_chunks', {'query': 'q'})]), LLMResponse(content='done')]
    supervisor = Supervisor(Scripted(responses), SessionStore(fakeredis.FakeRedis(), tmp_path / 'db.sqlite'), ResearchTools(retriever))
    supervisor.start('u', 's', 'q')
    event = supervisor.resume('u', 's')['tasks'][0]['calls'][0]
    assert retriever.calls == 2 and event['schema_retries'] == 1
    assert event['result'] == [{'id': 'v', 'text': 'valid'}] and event['status'] == 'completed'


def test_invalid_import_result_never_replays_write(tmp_path):
    writes = []
    def do_import(ident):
        writes.append(ident)
        return ['not a valid import receipt']
    plan = {'tasks': [{'id': 'i', 'worker': 'ingestion', 'goal': 'import', 'depends_on': []}]}
    responses = [LLMResponse(content=json.dumps(plan)), LLMResponse(tool_calls=[ToolCall('c', 'import_arxiv', {'id': '2401.00001'})]), LLMResponse(content='failed')]
    supervisor = Supervisor(Scripted(responses), SessionStore(fakeredis.FakeRedis(), tmp_path / 'db.sqlite'), ResearchTools(online_import=do_import))
    supervisor.start('u', 's', 'import')
    event = supervisor.resume('u', 's')['tasks'][0]['calls'][0]
    assert writes == ['2401.00001']
    assert event['status'] == 'failed' and '可能已经发生' in event['error']


def test_timed_out_retrieval_cancels_at_next_checkpoint(tmp_path, monkeypatch):
    import threading
    started = threading.Event()
    unblock = threading.Event()
    finished = threading.Event()
    class Slow:
        user = 'timeout-cancel'
        def retrieve_with_progress(self, query, progress):
            started.set()
            unblock.wait(3)
            try:
                progress('next', 'safe point')
                raise AssertionError('cancelled retrieval should not proceed')
            finally:
                finished.set()
    monkeypatch.setenv('CORECODER_SQLITE_PATH', str(tmp_path / 'db.sqlite'))
    monkeypatch.setenv('PAPER_RETRIEVAL_TIMEOUT', '0.03')
    monkeypatch.setenv('PAPER_RETRIEVAL_LOCK_WAIT', '0')
    tools = ResearchTools(Slow())
    try:
        with pytest.raises(TimeoutError, match='安全检查点'):
            tools.execute('retrieval', 'retrieve_chunks', {'query': 'q'})
        assert started.is_set()
        with pytest.raises(RuntimeError, match='后台运行'):
            tools.execute('retrieval', 'retrieve_chunks', {'query': 'q2'})
    finally:
        unblock.set()
    assert finished.wait(2)


def test_llm_retries_malformed_tool_json_without_executing(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from corecoder.llm import LLM
    def chunk(arguments):
        call = SimpleNamespace(index=0, id='c', function=SimpleNamespace(name='retrieve_chunks', arguments=arguments))
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(
            content=None, tool_calls=[call], reasoning_content=None))])
    llm = LLM(model='test', api_key='test')
    llm.client = Mock()
    llm.client.chat.completions.create.side_effect = [[chunk('{oops')], [chunk('{"query":"ok"}')]]
    monkeypatch.setattr('corecoder.llm.time.sleep', lambda _: None)
    schema = ResearchTools(Retriever()).schemas('retrieval')
    response = llm.chat(messages=[{'role': 'user', 'content': 'search'}], tools=schema)
    assert response.tool_calls[0].arguments == {'query': 'ok'}
    assert llm.client.chat.completions.create.call_count == 2
