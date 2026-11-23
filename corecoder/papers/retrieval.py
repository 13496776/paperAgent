"""BM25, embedding retrieval, reciprocal rank fusion and cross-encoder ranking."""
import math
import os
import re
from collections import Counter


def tokenize(text):
    # Chinese bigrams make unsegmented Chinese useful for lexical retrieval.
    words = re.findall(r"[a-zA-Z0-9]+|[\u3400-\u9fff]+", text.lower())
    tokens = []
    for word in words:
        if re.fullmatch(r"[\u3400-\u9fff]+", word) and len(word) > 1:
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word)
    return tokens


def bm25(query, chunks):
    documents = [tokenize(passage_text(c)) for c in chunks]
    counts = [Counter(doc) for doc in documents]
    average = sum(map(len, documents)) / len(documents) if documents else 0
    terms = tokenize(query)
    scores = []
    for doc, frequencies in zip(documents, counts):
        score = 0.0
        for term in terms:
            df = sum(term in item for item in counts)
            if df and average:
                tf = frequencies[term]
                score += math.log(1 + (len(documents) - df + .5) / (df + .5)) * tf * 2.5 / (
                    tf + 1.5 * (0.25 + 0.75 * len(doc) / average)
                )
        scores.append(score)
    return scores


def passage_text(chunk):
    section = chunk.get("section") or ""
    return "\n".join(part for part in (chunk.get("title", ""), section, chunk["text"]) if part)


class Retriever:
    def __init__(self, chunks, embedding_model, reranker):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.vectors = embedding_model.encode([passage_text(c) for c in chunks], normalize_embeddings=True) if chunks else []

    def retrieve(self, query, limit=None, candidates=None, progress=None):
        if limit is None:
            limit = int(os.environ.get("PAPER_RETRIEVAL_LIMIT", "5"))
        if candidates is None:
            candidates = int(os.environ.get("PAPER_RETRIEVAL_CANDIDATES", "30"))
        threshold = float(os.environ.get("PAPER_RERANK_THRESHOLD", "0.3"))
        if limit < 1 or candidates < 1 or not 0 <= threshold <= 1:
            raise ValueError("retrieval limits must be positive and rerank threshold must be between 0 and 1")
        if not self.chunks:
            return []
        if progress:
            progress("bm25", "关键词检索")
        lexical = bm25(query, self.chunks)
        if progress:
            progress("query_embedding", "编码查询向量")
        vector = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        if progress:
            progress("vector_search", "向量相似度检索")
        similarity = [sum(float(a) * float(b) for a, b in zip(vector, item)) for item in self.vectors]
        rankings = [sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidates]
                    for scores in (lexical, similarity)]
        ranked_lists = [[i for i in ranking] for ranking in rankings]
        return fuse_and_rerank(query, ranked_lists, {i: c for i, c in enumerate(self.chunks)},
                               self.reranker, limit, threshold, progress)


def fuse_and_rerank(query, rankings, chunks, reranker, limit, threshold, progress=None):
    if progress:
        progress("fusion", "融合并去重候选片段")
    fused = {}
    for ranking in rankings:
        for rank, ident in enumerate(ranking, 1):
            if ident in chunks:
                fused[ident] = fused.get(ident, 0) + 1 / (60 + rank)
    indexes = sorted(fused, key=fused.get, reverse=True)
    seen = set()
    unique = []
    for index in indexes:
        chunk = chunks[index]
        key = (chunk.get("paper_id") or chunk.get("id"), " ".join(chunk["text"].split()).casefold())
        if key not in seen:
            seen.add(key)
            unique.append(index)
    if not unique:
        return []
    if progress:
        progress("rerank", f"重排 {len(unique)} 个候选片段")
    scores = reranker.predict([(query, passage_text(chunks[i])) for i in unique],
                              activation_fn=lambda logits: logits.sigmoid())
    ranked = sorted(zip(unique, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [{**chunks[i], "rrf_score": fused[i], "rerank_score": float(score)}
            for i, score in ranked if float(score) >= threshold][:limit]
