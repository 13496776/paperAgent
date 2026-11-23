import json
import sqlite3

import pytest

pytest.importorskip("chromadb")
from corecoder.papers.storage import CorpusStore, _collection_name, _vector_id


class Encoder:
    def __init__(self, dims=2):
        self.dims = dims
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        return [[1.0] + [0.0] * (self.dims - 1) for _ in texts]


def paper(ident="p1"):
    return {"id": ident, "title": "研究论文", "authors": ["Alice"], "source": "Journal", "doi": "10.1/example",
            "ccf_rank": "A", "page_count": 2, "text": "# 方法\n第一部分检索方法。\f# 结果\n第二部分实验结果。"}


def test_persisted_vectors_metadata_keyword_and_user_isolation(tmp_path):
    path = tmp_path / "papers.sqlite"
    encoder = Encoder()
    store = CorpusStore(path)
    result = store.add("alice", paper(), encoder)
    assert result["page_count"] == 2 and result["ccf_rank"] == "A"
    assert result["status"] == "ready" and result["chunks"] == 2
    assert [c["page_start"] for c in store.chunks("alice")] == [1, 2]
    assert store.lexical_search("alice", "检索方法", 5)
    assert len(store.vector_search("alice", [1, 0], 5)) == 2
    assert store.vector_search("bob", [1, 0], 5) == []
    before = encoder.calls
    assert not store.add("alice", paper(), encoder)["created"]
    assert encoder.calls == before
    reopened = CorpusStore(path)
    assert len(reopened.vector_search("alice", [1, 0], 5)) == 2


def test_partial_index_is_recovered_without_duplicate_vectors(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path / "papers.sqlite")
    encoder = Encoder()
    original = store._collection
    failed = False

    class FailingOnce:
        def __init__(self, collection):
            self.collection = collection

        def upsert(self, **kwargs):
            nonlocal failed
            self.collection.upsert(**kwargs)
            if not failed:
                failed = True
                raise RuntimeError("interrupted after vector write")

        def __getattr__(self, name):
            return getattr(self.collection, name)

    monkeypatch.setattr(store, "_collection", lambda name: FailingOnce(original(name)))
    with pytest.raises(RuntimeError, match="interrupted"):
        store.add("alice", paper(), encoder)
    assert store.papers("alice")[0]["status"] == "indexing"
    assert store.chunks("alice") == []
    monkeypatch.setattr(store, "_collection", original)
    CorpusStore(store.path).ensure_indexed("alice", encoder)
    assert store.papers("alice")[0]["status"] == "ready"
    assert len(store.vector_search("alice", [1, 0], 10)) == 2


def test_delete_retries_and_never_deletes_other_users(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path / "papers.sqlite")
    encoder = Encoder()
    store.add("alice", paper(), encoder)
    store.add("bob", paper(), encoder)
    original = store._collection

    class FailingDelete:
        def __init__(self, collection):
            self.collection = collection

        def delete(self, **kwargs):
            self.collection.delete(**kwargs)
            raise RuntimeError("interrupted after deleting vectors")

    monkeypatch.setattr(store, "_collection", lambda name, path=None: FailingDelete(original(name, path)))
    with pytest.raises(RuntimeError, match="interrupted"):
        store.delete("alice", "p1")
    monkeypatch.setattr(store, "_collection", original)
    assert store.papers("alice") == []
    assert len(store.vector_search("bob", [1, 0], 5)) == 2
    assert store.papers("bob")[0]["title"] == "研究论文"
    assert not store.delete("alice", "p1")


def test_legacy_migration_is_repeatable_and_delete_does_not_resurrect(tmp_path):
    path = tmp_path / "legacy.sqlite"
    metadata = {"id": "old", "title": "Legacy", "authors": ["A"], "source": "J", "doi": ""}
    chunks = [{"id": "oldchunk", "paper_id": "old", "title": "Legacy", "text": "retrieval evidence"}]
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE paper_corpus(owner TEXT, id TEXT, metadata TEXT, chunks TEXT, PRIMARY KEY(owner,id))")
        db.execute("INSERT INTO paper_corpus VALUES (?, ?, ?, ?)", ("alice", "old", json.dumps(metadata), json.dumps(chunks)))
    store = CorpusStore(path)
    assert store.papers("alice")[0]["status"] == "indexing"
    CorpusStore(path).ensure_indexed("alice", Encoder())
    assert store.papers("alice")[0]["status"] == "ready"
    assert len(store.vector_search("alice", [1, 0], 5)) == 1
    assert store.delete("alice", "old")
    assert CorpusStore(path).papers("alice") == []
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM paper_corpus").fetchone()[0] == 0


def test_model_change_uses_separate_collection_and_delete_removes_both(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EMBED_MODEL", "model-a")
    store = CorpusStore(tmp_path / "papers.sqlite")
    store.add("alice", paper(), Encoder())
    first = _collection_name("model-a")
    monkeypatch.setenv("PAPER_EMBED_MODEL", "model-b")
    updated = CorpusStore(store.path)
    updated.ensure_indexed("alice", Encoder(3))
    assert updated.papers("alice")[0]["embedding_model"] == "model-b"
    assert len(updated.vector_search("alice", [1, 0, 0], 5)) == 2
    second = _collection_name("model-b")
    assert updated.delete("alice", "p1")
    for name in (first, second):
        assert not updated._collection(name).get(where={"owner": "alice"}, include=[])["ids"]


def test_pdf_page_numbers_preserve_leading_blank_page():
    from corecoder.papers.corpus import chunks_for_paper

    chunks = chunks_for_paper({"id": "p", "title": "Paper", "page_count": 2, "text": "\fText on page two"})
    assert chunks[0]["page_start"] == 2
    assert _vector_id("alice", "p", chunks[0]["id"]) != _vector_id("bob", "p", chunks[0]["id"])


def test_delete_while_indexing_cannot_publish_obsolete_vectors(tmp_path):
    import threading

    started = threading.Event()
    release = threading.Event()
    store = CorpusStore(tmp_path / "race.sqlite")

    class BlockingEncoder(Encoder):
        def encode(self, texts, normalize_embeddings=True):
            started.set()
            assert release.wait(5)
            return super().encode(texts, normalize_embeddings)

    errors = []
    def add_old():
        try:
            store.add("alice", paper(), BlockingEncoder())
        except ValueError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=add_old)
    thread.start()
    try:
        assert started.wait(5)
        assert store.delete("alice", "p1")
        replacement = paper()
        replacement["text"] = "replacement document"
        replacement["page_count"] = 1
        store.add("alice", replacement, Encoder())
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert [chunk["text"] for chunk in store.chunks("alice")] == ["replacement document"]
    assert len(store.vector_search("alice", [1, 0], 10)) == 1


def test_vectors_survive_process_restart(tmp_path):
    import os
    import subprocess
    import sys

    path = tmp_path / "restart.sqlite"
    env = os.environ.copy()
    env.pop("PAPER_CHROMA_PATH", None)
    env["PAPER_EMBED_MODEL"] = "restart-test-model"
    create = '''
from corecoder.papers.storage import CorpusStore
class E:
    def encode(self, texts, normalize_embeddings=True): return [[1.0, 0.0] for _ in texts]
CorpusStore(PATH).add("alice", {"id":"p", "title":"Paper", "text":"retrieval evidence", "authors":"", "source":"", "doi":""}, E())
'''.replace("PATH", repr(str(path)))
    query = '''
from corecoder.papers.storage import CorpusStore
store = CorpusStore(PATH)
assert len(store.vector_search("alice", [1.0, 0.0], 5)) == 1
assert len(store.lexical_search("alice", "retrieval", 5)) == 1
'''.replace("PATH", repr(str(path)))
    subprocess.run([sys.executable, "-c", create], check=True, env=env, timeout=30)
    subprocess.run([sys.executable, "-c", query], check=True, env=env, timeout=30)


def test_shared_json_vectors_are_persistent_and_reused(tmp_path):
    path = tmp_path / "shared.sqlite"
    store = CorpusStore(path)
    chunks = [{"id": "c1", "paper_id": "p1", "title": "Paper", "text": "retrieval"},
              {"id": "c2", "paper_id": "p2", "title": "Paper 2", "text": "evidence"}]
    encoder = Encoder()
    assert len(store.shared_vector_search(chunks, [1, 0], encoder, 5)) == 2
    assert encoder.calls == 1
    assert len(CorpusStore(path).shared_vector_search(chunks, [1, 0], encoder, 5)) == 2
    assert encoder.calls == 1
    chunks.append({"id": "c3", "paper_id": "p3", "title": "Paper 3", "text": "new"})
    assert len(store.shared_vector_search(chunks, [1, 0], encoder, 5)) == 3
    assert encoder.calls == 2


def test_metadata_preserves_ccf_provenance_and_validation(tmp_path):
    store = CorpusStore(tmp_path / "metadata.sqlite")
    data = paper()
    data.update(venue="Conference", year=2024, ccf_source="CCF list", ccf_version="2022", ccf_verified=True)
    result = store.add("alice", data, Encoder())
    assert (result["venue"], result["year"], result["ccf_source"], result["ccf_version"], result["ccf_verified"]) == (
        "Conference", 2024, "CCF list", "2022", True)
    bad = paper("p2")
    bad.update(ccf_verified=True)
    with pytest.raises(ValueError, match="source and version"):
        store.add("alice", bad, Encoder())


def test_chroma_path_change_reindexes_and_deletes_both_locations(tmp_path, monkeypatch):
    path = tmp_path / "path-change.sqlite"
    first_path = tmp_path / "index-a"
    second_path = tmp_path / "index-b"
    monkeypatch.setenv("PAPER_CHROMA_PATH", str(first_path))
    first = CorpusStore(path)
    first.add("alice", paper(), Encoder())
    monkeypatch.setenv("PAPER_CHROMA_PATH", str(second_path))
    second = CorpusStore(path)
    second.ensure_indexed("alice", Encoder())
    assert len(second.vector_search("alice", [1, 0], 5)) == 2
    assert second.delete("alice", "p1")
    name = _collection_name(second.model)
    assert first._collection(name).get(where={"owner": "alice"}, include=[])["ids"] == []
    assert second._collection(name).get(where={"owner": "alice"}, include=[])["ids"] == []


def test_concurrent_duplicate_import_creates_one_paper(tmp_path):
    import threading

    store = CorpusStore(tmp_path / "concurrent.sqlite")
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def upload():
        try:
            barrier.wait(timeout=5)
            results.append(store.add("alice", paper(), Encoder()))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=upload) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert sorted(result["created"] for result in results) == [False, True]
    assert len(store.papers("alice")) == 1
    assert len(store.vector_search("alice", [1, 0], 10)) == 2
