import io
import time

import pytest

pytest.importorskip("pypdf")
pytest.importorskip("chromadb")
pytest.importorskip("fastapi")
jwt = pytest.importorskip("jwt")
pytest.importorskip("redis")
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from corecoder import web
from corecoder.papers.ingestion import CorpusStore, extract_pdf, parse_arxiv_feed


def make_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (A paper about retrieval.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": "Research Paper", "/Author": "Example Author"})
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class Embedding:
    def encode(self, texts, normalize_embeddings=True):
        return [[1.0, 0.0] for _ in texts]


def test_pdf_import_is_deduplicated_and_user_scoped(tmp_path):
    paper = extract_pdf(make_pdf(), "paper.pdf")
    assert "retrieval" in paper["text"]
    assert paper["title"] == "Research Paper"
    store = CorpusStore(tmp_path / "corpus.sqlite")
    assert store.add("alice", paper, Embedding())["created"]
    assert not store.add("alice", paper, Embedding())["created"]
    assert len(store.chunks("alice")) == 1
    assert store.chunks("bob") == []
    with pytest.raises(ValueError):
        extract_pdf(b"not a pdf", "bad.pdf")


def test_arxiv_feed_metadata():
    feed = b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <entry><id>http://arxiv.org/abs/2401.12345v1</id><title>Paper title</title><summary>Abstract</summary>
    <author><name>Researcher</name></author><arxiv:doi>10.1/example</arxiv:doi></entry></feed>'''
    result = parse_arxiv_feed(feed)[0]
    assert result["id"] == "2401.12345v1"
    assert result["doi"] == "10.1/example"
    assert result["authors"] == ["Researcher"]


def test_authenticated_upload_and_list(tmp_path, monkeypatch):
    key = "test-key-that-is-long-enough-for-hs256"
    monkeypatch.setenv("CORECODER_JWT_KEY", key)
    monkeypatch.setenv("CORECODER_JWT_ISSUER", "issuer")
    monkeypatch.setenv("CORECODER_JWT_AUDIENCE", "app")
    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "test.sqlite"))
    class Reranker:
        def predict(self, pairs, activation_fn):
            return [0.9 for _ in pairs]

    monkeypatch.setattr(web, "get_models", lambda: (Embedding(), Reranker()))
    client = TestClient(web.app)
    now = int(time.time())
    def auth(user):
        token = jwt.encode({"sub": user, "iss": "issuer", "aud": "app", "iat": now, "exp": now + 60},
                           key, algorithm="HS256")
        return {"Authorization": "Bearer " + token}
    response = client.post("/api/corpus/upload", files={"file": ("paper.pdf", make_pdf(), "application/pdf")},
                           data={"ccf_rank": "A"}, headers=auth("alice"))
    assert response.status_code == 200, response.text
    assert response.json()["chunks"] == 1
    assert response.json()["ccf_rank"] == "A"
    assert response.json()["page_count"] == 1
    paper_id = response.json()["id"]
    assert len(client.get("/api/corpus", headers=auth("alice")).json()) == 1
    assert client.get("/api/corpus", headers=auth("bob")).json() == []
    assert web.UserRetriever("alice").retrieve("retrieval")[0]["title"] == "Research Paper"
    assert web.UserRetriever("bob").retrieve("retrieval") == []
    assert client.delete(f"/api/corpus/{paper_id}", headers=auth("bob")).status_code == 404
    assert client.delete(f"/api/corpus/{paper_id}", headers=auth("alice")).status_code == 200
    assert web.UserRetriever("alice").retrieve("retrieval") == []
    assert client.get("/api/corpus").status_code in (401, 403)


def test_arxiv_import_with_mocked_remote(tmp_path):
    import httpx

    from corecoder.papers.ingestion import get_arxiv_paper, search_arxiv

    feed = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2401.12345v1</id>
    <title>Example</title><summary>Research abstract</summary><author><name>A</name></author></entry></feed>'''
    def response(request):
        if request.url.host == "export.arxiv.org":
            return httpx.Response(200, content=feed)
        assert request.url.host == "arxiv.org"
        return httpx.Response(200, content=make_pdf())
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        assert search_arxiv("retrieval", client)[0]["title"] == "Example"
        paper = get_arxiv_paper("2401.12345v1", client)
    assert paper["title"] == "Example"
    assert paper["source"] == "https://arxiv.org/abs/2401.12345v1"
    assert CorpusStore(tmp_path / "test.sqlite").add("alice", paper, Embedding())["created"]
    with pytest.raises(ValueError):
        get_arxiv_paper("../../bad")


def test_inventory_and_retriever_cache_with_corpus_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("PAPER_CORPUS_PATH", raising=False)

    class CountingEmbedding:
        def __init__(self):
            self.batches = []

        def encode(self, texts, normalize_embeddings=True):
            self.batches.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    class Reranker:
        def predict(self, pairs, activation_fn):
            return [0.9 for _ in pairs]

    embedding = CountingEmbedding()
    reranker = Reranker()
    monkeypatch.setattr(web, "get_models", lambda: (embedding, reranker))
    store = web.corpus_store()
    for i in range(3):
        store.add("alice", {"id": str(i), "title": f"Paper {i}", "text": f"retrieval {i}",
                            "authors": "", "source": "", "doi": ""}, embedding)
    retriever = web.UserRetriever("alice")
    assert retriever.list_papers()["count"] == 3
    assert embedding.batches == [1, 1, 1]
    assert len(retriever.retrieve("retrieval")) == 3
    assert len(retriever.retrieve("retrieval")) == 3
    assert embedding.batches == [1, 1, 1, 1, 1]
    store.add("alice", {"id": "3", "title": "Paper 3", "text": "retrieval 3", "authors": "", "source": "", "doi": ""}, embedding)
    assert retriever.list_papers()["count"] == 4
    assert len(retriever.retrieve("retrieval")) == 4
    assert embedding.batches[-2:] == [1, 1]
    assert web.UserRetriever("bob").list_papers()["count"] == 0


def test_inventory_includes_shared_index_once(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "shared.sqlite"))
    index = tmp_path / "shared.json"
    index.write_text(json.dumps([{"paper_id": "shared", "title": "Paper", "text": "part 1"},
                                 {"paper_id": "shared", "title": "Paper", "text": "part 2"}]), encoding="utf-8")
    monkeypatch.setenv("PAPER_CORPUS_PATH", str(index))
    assert web.UserRetriever("alice").list_papers()["count"] == 1


def test_rag_workflow_with_real_retrieval_components_and_mocked_models(tmp_path, monkeypatch):
    from corecoder.llm import LLMResponse, ToolCall
    from corecoder.persistent import SessionStore
    from corecoder.workflow.engine import Supervisor

    fakeredis = pytest.importorskip("fakeredis")
    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "rag.sqlite"))
    monkeypatch.delenv("PAPER_CORPUS_PATH", raising=False)

    class Reranker:
        def predict(self, pairs, activation_fn):
            return [0.9 for _ in pairs]

    embedding, reranker = Embedding(), Reranker()
    monkeypatch.setattr(web, "get_models", lambda: (embedding, reranker))
    web.corpus_store().add("alice", {"id": "p1", "title": "Retrieval paper", "text": "RAG uses retrieved evidence.",
                                    "authors": "A", "source": "Journal", "doi": "10.1000/example"}, embedding)

    class Model:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content='{"tasks":[{"id":"a","worker":"retrieval","goal":"检索RAG","depends_on":[]},'
                                           '{"id":"b","worker":"analysis","goal":"回答","depends_on":["a"]}]}')
            if self.calls == 2:
                return LLMResponse(tool_calls=[ToolCall("t1", "retrieve_chunks", {"query": "RAG evidence"})])
            if self.calls == 3:
                import json
                hits = json.loads(messages[-1]["content"])
                assert hits[0]["title"] == "Retrieval paper"
                assert hits[0]["doi"] == "10.1000/example"
                return LLMResponse(content="RAG uses evidence [Retrieval paper].")
            assert "RAG uses evidence" in messages[-1]["content"]
            assert "10.1000/example" in messages[-1]["content"]
            return LLMResponse(content="Answer citing [Retrieval paper].")

    store = SessionStore(fakeredis.FakeRedis(), tmp_path / "sessions.sqlite")
    supervisor = Supervisor(Model(), store, web.get_tools("alice"))
    supervisor.start("alice", "rag", "What does RAG use?")
    result = supervisor.resume("alice", "rag")
    assert result["status"] == "completed"
    assert "Retrieval paper" in result["result"]
    stages = [event["stage"] for event in result["tasks"][0]["activities"]]
    assert all(stage in stages for stage in ("model", "corpus_load", "bm25", "query_embedding", "vector_search", "rerank"))


def test_rag_uses_openai_compatible_streaming_and_tool_results(tmp_path, monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from corecoder.llm import LLM
    from corecoder.persistent import SessionStore
    from corecoder.workflow.engine import Supervisor

    fakeredis = pytest.importorskip("fakeredis")
    requests = []
    plan = '{"tasks":[{"id":"a","worker":"retrieval","goal":"find evidence","depends_on":[]},'
    plan += '{"id":"b","worker":"analysis","goal":"answer","depends_on":["a"]}]}'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if len(requests) == 1:
                delta = {"role": "assistant", "content": plan}
                finish = "stop"
            elif len(requests) == 2:
                delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": "t1", "type": "function",
                         "function": {"name": "retrieve_chunks", "arguments": '{"query":"RAG evidence"}'}}]}
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "Evidence from Retrieval paper."}
                finish = "stop"
            chunk = {"id": "chat-test", "object": "chat.completion.chunk", "created": 1, "model": "test",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            payload = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    class Ranker:
        def predict(self, pairs, activation_fn):
            return [0.9 for _ in pairs]

    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "corpus.sqlite"))
    monkeypatch.delenv("PAPER_CORPUS_PATH", raising=False)
    encoder, ranker = Embedding(), Ranker()
    monkeypatch.setattr(web, "get_models", lambda: (encoder, ranker))
    web.corpus_store().add("alice", {"id": "p1", "title": "Retrieval paper", "text": "RAG uses evidence.",
                                    "authors": "A", "source": "Journal", "doi": "10.1000/example"}, encoder)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = LLM("test", "test-key", base_url=f"http://127.0.0.1:{server.server_port}/v1", timeout=5,
                    client_max_retries=0)
        store = SessionStore(fakeredis.FakeRedis(), tmp_path / "session.sqlite")
        supervisor = Supervisor(model, store, web.get_tools("alice"))
        supervisor.start("alice", "session", "What evidence does RAG use?")
        result = supervisor.resume("alice", "session")
        assert result["status"] == "completed"
        assert len(requests) == 4
        assert any(message.get("role") == "tool" and "10.1000/example" in message["content"]
                   for message in requests[2]["messages"])
        assert "10.1000/example" in requests[3]["messages"][-1]["content"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_deleted_paper_is_not_returned_by_inflight_rerank(tmp_path, monkeypatch):
    monkeypatch.setenv("CORECODER_SQLITE_PATH", str(tmp_path / "race.sqlite"))
    monkeypatch.delenv("PAPER_CORPUS_PATH", raising=False)
    encoder = Embedding()
    store = web.corpus_store()
    store.add("alice", {"id": "p1", "title": "Paper", "text": "retrieval evidence", "authors": "", "source": "", "doi": ""}, encoder)

    class DeletingReranker:
        def predict(self, pairs, activation_fn):
            assert store.delete("alice", "p1")
            return [0.9 for _ in pairs]

    monkeypatch.setattr(web, "get_models", lambda: (encoder, DeletingReranker()))
    assert web.UserRetriever("alice").retrieve("retrieval") == []
