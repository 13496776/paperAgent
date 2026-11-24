"""Authenticated, checkpointed sessions for the existing tool-calling agent."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import jwt


@dataclass(frozen=True)
class JWTAuthenticator:
    key: str
    issuer: str
    audience: str
    algorithms: tuple[str, ...] = ("HS256",)

    def authenticate(self, token: str) -> str:
        claims = jwt.decode(token, self.key, algorithms=list(self.algorithms), issuer=self.issuer,
                            audience=self.audience, options={"require": ["sub", "exp", "iat", "iss", "aud"]})
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("JWT subject must be a nonempty user ID")
        return subject


SESSION_TTL_SECONDS = 24 * 60 * 60


class SessionStore:
    """Redis checkpoints and SQLite summaries and completed workflows.

    Redis retains active history and workflow state for 24 hours since the
    last read or write. SQLite retains summaries and completed results.
    """

    def __init__(self, redis, sqlite_path):
        self.redis = redis
        self.sqlite_path = str(sqlite_path)
        self.results_path = Path(self.sqlite_path).absolute().parent / "workflow_results"
        with sqlite3.connect(self.sqlite_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS summaries (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                       "session TEXT NOT NULL, summary TEXT NOT NULL, created REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS completed_workflows ("
                       "owner TEXT NOT NULL, session TEXT NOT NULL, workflow_id TEXT NOT NULL, "
                       "revision INTEGER NOT NULL, workflow TEXT NOT NULL, PRIMARY KEY (owner, session, workflow_id))")
            db.execute("CREATE INDEX IF NOT EXISTS completed_workflows_revision "
                       "ON completed_workflows (owner, session, revision DESC)")
            archived = db.execute("SELECT owner, session, workflow FROM completed_workflows").fetchall()
        for owner, session, workflow in archived:
            self.save_workflow_json(owner, session, json.loads(workflow), only_if_missing=True)

    @staticmethod
    def _key(user, session):
        if not isinstance(session, str) or not session.strip():
            raise ValueError("session ID must be nonempty")
        digest = hashlib.sha256(json.dumps([user, session], ensure_ascii=False).encode()).hexdigest()
        return f"corecoder:session:{digest}"

    def load(self, user, session):
        key = self._key(user, session)
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.get(key)
            pipe.expire(key, SESSION_TTL_SECONDS)
            data, _ = pipe.execute()
        return json.loads(data) if data else {"messages": [], "summary_id": None, "status": "idle"}

    def save(self, user, session, state, lock):
        from redis.exceptions import WatchError

        with self.redis.pipeline() as pipe:
            try:
                pipe.watch(lock.name)
                if pipe.get(lock.name) != lock.local.token:
                    raise RuntimeError("session lock lost; refusing to save")
                pipe.multi()
                pipe.set(self._key(user, session), json.dumps(state, ensure_ascii=False), ex=SESSION_TTL_SECONDS)
                pipe.execute()
            except WatchError as exc:
                raise RuntimeError("session lock changed; refusing to save") from exc

    def archive(self, user, session, summary):
        snapshot = uuid.uuid4().hex
        with sqlite3.connect(self.sqlite_path) as db:
            db.execute("INSERT INTO summaries VALUES (?, ?, ?, ?, ?)", (snapshot, user, session, summary, time.time()))
        return snapshot

    def save_completed_workflow(self, user, session, workflow):
        with sqlite3.connect(self.sqlite_path) as db:
            db.execute(
                "INSERT INTO completed_workflows (owner, session, workflow_id, revision, workflow) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (owner, session, workflow_id) DO UPDATE SET revision=excluded.revision, workflow=excluded.workflow "
                "WHERE excluded.revision > completed_workflows.revision",
                (user, session, workflow["id"], workflow["revision"], json.dumps(workflow, ensure_ascii=False)),
            )

    def save_workflow_json(self, user, session, workflow, only_if_missing=False):
        identity = json.dumps([user, session, workflow["id"]], ensure_ascii=False).encode("utf-8")
        filename = hashlib.sha256(identity).hexdigest() + ".json"
        self.results_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = self.results_path / filename
        if only_if_missing and target.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=".workflow-", suffix=".tmp", dir=self.results_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({"user": user, "session": session, "workflow": workflow}, output, ensure_ascii=False, indent=2)
                output.write("\n")
            if only_if_missing:
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            else:
                os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def latest_completed_workflow(self, user, session):
        with sqlite3.connect(self.sqlite_path) as db:
            row = db.execute(
                "SELECT workflow FROM completed_workflows WHERE owner=? AND session=? ORDER BY revision DESC LIMIT 1",
                (user, session),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def summary(self, user, session, snapshot):
        if not snapshot:
            return ""
        with sqlite3.connect(self.sqlite_path) as db:
            row = db.execute("SELECT summary FROM summaries WHERE id=? AND owner=? AND session=?",
                             (snapshot, user, session)).fetchone()
        if row is None:
            raise RuntimeError("session summary is missing")
        return row[0]


class PersistentAgent:
    """Run an Agent with checkpoints after every message and tool receipt."""

    def __init__(self, agent, store, authenticator, window=12, completion_check=None):
        if window < 2:
            raise ValueError("window must be at least 2")
        self.agent = agent
        self.store = store
        self.authenticator = authenticator
        self.window = window
        self.completion_check = completion_check

    def run(self, token, session, message=None):
        user = self.authenticator.authenticate(token)
        key = self.store._key(user, session)
        lock = self.store.redis.lock(key + ":lock", timeout=3600, blocking_timeout=0)
        if not lock.acquire():
            raise RuntimeError("session is already running")
        try:
            state = self.store.load(user, session)
            self.agent.messages = state["messages"]
            # An interrupted call may have performed external side effects.
            # Never replay it automatically when no durable receipt exists.
            self._recover(state, user, session, lock)
            if message is not None:
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("message must be a nonempty string")
                self.agent.messages.append({"role": "user", "content": message})
                state["status"] = "running"
                self.store.save(user, session, state, lock)
            elif not self.agent.messages:
                raise ValueError("no session history to resume")
            summary = self.store.summary(user, session, state.get("summary_id"))
            for _ in range(self.agent.max_rounds):
                messages = self.agent._full_messages()
                if summary:
                    messages.insert(1, {"role": "system", "content": "Earlier conversation summary:\n" + summary})
                response = self.agent.llm.chat(messages=messages, tools=self.agent._tool_schemas())
                self.agent.messages.append(response.message)
                state["status"] = "running"
                self.store.save(user, session, state, lock)
                if not response.tool_calls:
                    feedback = self.completion_check(response.content, state) if self.completion_check else None
                    if feedback:
                        self.agent.messages.append({"role": "user", "content": str(feedback)})
                        self.store.save(user, session, state, lock)
                        continue
                    state["status"] = "completed"
                    self._compress(state, user, session, lock)
                    self.store.save(user, session, state, lock)
                    return response.content
                for call in response.tool_calls:
                    result = self.agent._pre_hooks(call) or self.agent._permit(call)
                    if result is None:
                        result = self.agent._exec_tool(call)
                        self.agent._post_hooks(call, result)
                    self.agent.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    self.store.save(user, session, state, lock)
            state["status"] = "round_limit"
            self.store.save(user, session, state, lock)
            return "(reached maximum tool-call rounds)"
        finally:
            if lock.owned():
                lock.release()

    def _recover(self, state, user, session, lock):
        pending = {}
        for entry in self.agent.messages:
            if entry.get("role") == "assistant":
                for call in entry.get("tool_calls") or []:
                    pending[call["id"]] = call
            elif entry.get("role") == "tool":
                pending.pop(entry.get("tool_call_id"), None)
        for call in pending.values():
            self.agent.messages.append({"role": "tool", "tool_call_id": call["id"],
                                        "content": "[interrupted: tool outcome unknown; do not assume it failed]"})
        if pending:
            state["status"] = "interrupted"
            self.store.save(user, session, state, lock)

    def _compress(self, state, user, session, lock):
        messages = self.agent.messages
        if len(messages) <= self.window:
            return
        split = self.agent.context._safe_split(messages, self.window)
        if not split:
            return
        previous = self.store.summary(user, session, state.get("summary_id"))
        old = messages[:split]
        response = self.agent.llm.chat(messages=[
            {"role": "system", "content": "Summarize the prior conversation. Preserve decisions, research findings, "
                                          "citations, uncertainty, and unfinished tasks."},
            {"role": "user", "content": json.dumps({"previous_summary": previous, "messages": old}, ensure_ascii=False)},
        ])
        if not response.content:
            raise ValueError("empty conversation summary")
        state["summary_id"] = self.store.archive(user, session, response.content)
        del messages[:split]
