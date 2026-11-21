import json
import os

import pytest

from corecoder.papers.corpus import chunks_for_paper
from corecoder.papers.retrieval import Retriever
from corecoder.runtime_config import apply_runtime_config


def test_config_applies_to_chunks_and_respects_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"chunk_size": 100, "chunk_overlap": 10, "embedding_model": "local-model",
                                  "deepseek_api_key": "private"}), encoding="utf-8")
    for name in ("PAPER_CHUNK_SIZE", "PAPER_CHUNK_OVERLAP", "PAPER_EMBED_MODEL", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PAPER_CHUNK_SIZE", "80")
    apply_runtime_config(config)
    assert os.environ["PAPER_EMBED_MODEL"] == "local-model"
    chunks = chunks_for_paper({"title": "Test", "text": "a" * 120})
    assert [len(chunk["text"]) for chunk in chunks] == [80, 50]
    assert os.environ["DEEPSEEK_API_KEY"] == "private"


def test_config_rejects_unknown_field(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"unexpected": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        apply_runtime_config(config)


def test_configured_rerank_threshold(monkeypatch):
    class Embedding:
        def encode(self, texts, normalize_embeddings=True):
            return [[1.0] for _ in texts]

    class Ranker:
        def predict(self, pairs, activation_fn):
            return [0.6]

    monkeypatch.setenv("PAPER_RERANK_THRESHOLD", "0.7")
    chunk = {"title": "Paper", "text": "research"}
    assert Retriever([chunk], Embedding(), Ranker()).retrieve("research") == []
    monkeypatch.setenv("PAPER_RERANK_THRESHOLD", "0.5")
    assert len(Retriever([chunk], Embedding(), Ranker()).retrieve("research")) == 1
