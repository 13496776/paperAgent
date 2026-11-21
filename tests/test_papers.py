import json

from corecoder.papers.agent import PaperAgent
from corecoder.papers.corpus import build_corpus
from corecoder.papers.evaluation import evaluate
from corecoder.papers.retrieval import Retriever, bm25


class Embedding:
    def encode(self, texts, normalize_embeddings=True):
        return [[1.0, 0.0] if "retrieval" in text else [0.0, 1.0] for text in texts]


class Reranker:
    def predict(self, pairs, activation_fn=None):
        return [1.0 if "retrieval" in text else 0.0 for _, text in pairs]


class LLM:
    def chat(self, messages):
        class Response:
            content = ""
        result = Response()
        system = messages[0]["content"]
        if "string fields" in system:
            result.content = json.dumps({"intent": "search", "resolved_question": "retrieval", "query": "retrieval"})
        elif "Assess" in system:
            result.content = '{"supported": true}'
        elif "Summarize" in system:
            result.content = "Previous research on retrieval."
        else:
            result.content = "The paper studies retrieval [P1]."
        return result


def test_pipeline_and_memory(tmp_path):
    path = tmp_path / "papers.json"
    index = tmp_path / "index.json"
    path.write_text(json.dumps([{"title": "Research", "authors": ["A"], "doi": "10/test", "source": "Journal", "text": "# Methods\nretrieval works."}]))
    assert build_corpus(path, index, embedding_model=Embedding()) == 1
    chunks = json.loads(index.read_text())
    assert chunks[0]["section"] == "Methods"
    assert bm25("retrieval", chunks)[0] > 0
    agent = PaperAgent(LLM(), Retriever(chunks, Embedding(), Reranker()), window=1)
    answer = agent.chat("What works?")
    assert "[P1]" in answer and "10/test" in answer
    agent.chat("And that?")
    assert len(agent.state.turns) == 1
    assert agent.state.summary
    assert [s["stage"] for s in agent.state.pipeline_trace[0]["stages"]] == [
        "query_analysis", "retrieval", "evidence_reflection", "answer"
    ]
    assert agent.state.pipeline_trace[0]["stages"][1]["tool_calls"][0]["name"] == "retrieve_chunks"


def test_metrics():
    result = evaluate([{"intent": "search", "expected_intent": "search", "query": "test", "expected_query": "test",
                        "relevant_ids": ["a", "b"], "retrieved_ids": ["a"]}])
    assert result["intent_accuracy"] == 1
    assert result["rewrite_exact_match"] == 1
    assert result["retrieval_recall"] == 0.5


def test_headings_and_page_margins():
    from corecoder.papers.corpus import chunks_for_paper

    text = ("Journal of Studies\n# Methods\nMethod text.\n\n## Training\nTraining text.\nPage 1"
            "\fJournal of Studies\n## Testing\nTesting text.\nPage 2")
    chunks = chunks_for_paper({"title": "Study", "text": text})
    assert [chunk["section"] for chunk in chunks] == ["Methods", "Methods > Training", "Methods > Testing"]
    assert all("Journal of Studies" not in chunk["text"] and "Page " not in chunk["text"] for chunk in chunks)


def test_semantic_boundaries_and_overlap():
    from corecoder.papers.corpus import chunks_for_paper

    class ParagraphEncoder:
        def encode(self, texts, normalize_embeddings=True):
            return [[1, 0] if text.startswith("A") else [0, 1] for text in texts]

    text = "A" * 500 + "\n\n" + "B" * 600
    chunks = chunks_for_paper({"title": "Study", "text": text}, embedding_model=ParagraphEncoder())
    assert len(chunks) == 2
    assert all(len(chunk["text"]) <= 1024 for chunk in chunks)
    assert chunks[0]["text"].endswith("A" * 48)
    assert chunks[1]["text"].startswith("A" * 48)
    assert chunks[1]["text"].endswith("B" * 600)


def test_reranker_score_threshold_and_heading_context():
    class Scores:
        def predict(self, pairs, activation_fn):
            assert all("Parent > Child" in passage for _, passage in pairs)
            return [0.29, 0.3, 0.91]

    chunks = [{"title": "Paper", "section": "Parent > Child", "text": str(i), "id": str(i)} for i in range(3)]
    results = Retriever(chunks, Embedding(), Scores()).retrieve("query")
    assert [hit["id"] for hit in results] == ["2", "1"]
    assert [hit["rerank_score"] for hit in results] == [0.91, 0.3]
