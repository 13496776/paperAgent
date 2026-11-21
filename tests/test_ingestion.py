import io
import time

import pytest

pytest.importorskip("pypdf")
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
                           headers=auth("alice"))
    assert response.status_code == 200, response.text
    assert response.json()["chunks"] == 1
    assert len(client.get("/api/corpus", headers=auth("alice")).json()) == 1
    assert client.get("/api/corpus", headers=auth("bob")).json() == []
    assert web.UserRetriever("alice").retrieve("retrieval")[0]["title"] == "Research Paper"
    assert web.UserRetriever("bob").retrieve("retrieval") == []
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
