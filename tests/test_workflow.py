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
