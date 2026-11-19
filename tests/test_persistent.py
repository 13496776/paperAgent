import time
from typing import ClassVar

import pytest

fakeredis = pytest.importorskip("fakeredis")
jwt = pytest.importorskip("jwt")
pytest.importorskip("redis")

from corecoder.agent import Agent
from corecoder.llm import LLMResponse, ToolCall
from corecoder.persistent import JWTAuthenticator, PersistentAgent, SessionStore
from corecoder.tools.base import Tool

KEY = "test-key-with-at-least-32-bytes-long"


def token(subject="alice"):
    return jwt.encode({"sub": subject, "iss": "issuer", "aud": "app", "iat": int(time.time()),
                       "exp": int(time.time()) + 1000}, KEY, algorithm="HS256")


class ToolForTest(Tool):
    name = "example"
    description = "Example"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def execute(self):
        return "receipt"


class Scripted:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, **kwargs):
        return next(self.responses)


def setup(tmp_path, responses, window=12):
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "memory.sqlite")
    agent = Agent(Scripted(responses), tools=[ToolForTest()])
    auth = JWTAuthenticator(KEY, "issuer", "app")
    return store, PersistentAgent(agent, store, auth, window=window)


def test_tool_loop_and_user_isolation(tmp_path):
    call = ToolCall(id="call-1", name="example", arguments={})
    store, agent = setup(tmp_path, [LLMResponse(tool_calls=[call]), LLMResponse(content="done")])
    assert agent.run(token(), "session", "question") == "done"
    state = store.load("alice", "session")
    assert [item["role"] for item in state["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert state["messages"][2]["content"] == "receipt"
    assert store.load("bob", "session")["messages"] == []
    with pytest.raises(jwt.InvalidTokenError):
        agent.run("invalid", "session", "question")


def test_recovery_does_not_replay_unknown_tool(tmp_path):
    store, agent = setup(tmp_path, [LLMResponse(content="recovered")])
    call = ToolCall(id="pending", name="example", arguments={})
    state = {"messages": [{"role": "user", "content": "question"}, LLMResponse(tool_calls=[call]).message],
             "summary_id": None, "status": "running"}
    lock = store.redis.lock(store._key("alice", "session") + ":lock", timeout=60)
    assert lock.acquire()
    store.save("alice", "session", state, lock)
    lock.release()
    assert agent.run(token(), "session") == "recovered"
    history = store.load("alice", "session")["messages"]
    assert "outcome unknown" in history[2]["content"]


def test_compression_stores_summary(tmp_path):
    store, agent = setup(tmp_path, [LLMResponse(content="one"), LLMResponse(content="two"),
                                   LLMResponse(content="older conversation")], window=2)
    agent.run(token(), "session", "first")
    agent.run(token(), "session", "second")
    state = store.load("alice", "session")
    assert len(state["messages"]) == 2
    assert store.summary("alice", "session", state["summary_id"]) == "older conversation"


def test_session_ttl_is_set_and_refreshed(tmp_path):
    store, agent = setup(tmp_path, [LLMResponse(content="done")])
    agent.run(token(), "session", "question")
    key = store._key("alice", "session")
    assert 86398 <= store.redis.ttl(key) <= 86400

    store.redis.expire(key, 5)
    assert store.load("alice", "session")["messages"]
    assert 86398 <= store.redis.ttl(key) <= 86400

    # A legacy session without an expiry receives one when accessed.
    store.redis.persist(key)
    assert store.redis.ttl(key) == -1
    store.load("alice", "session")
    assert 86398 <= store.redis.ttl(key) <= 86400

    store.redis.delete(key)
    assert store.load("alice", "session")["messages"] == []
    assert store.redis.ttl(key) == -2
