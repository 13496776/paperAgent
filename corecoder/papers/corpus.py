"""Local JSON corpus and structure-aware, semantic passage splitting."""
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(os.environ.get("PAPER_EMBED_MODEL", DEFAULT_EMBED_MODEL))


def remove_page_margins(text):
    """Remove repeated first/last lines on form-feed-delimited pages."""
    pages = text.split("\f")
    if len(pages) < 2:
        return text
    page_lines = [page.splitlines() for page in pages]
    edge_lines = []
    counts = Counter()
    for lines in page_lines:
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edges = set(nonempty[:1] + nonempty[-1:])
        edge_lines.append(edges)
        keys = {re.sub(r"\d+", "#", lines[i].strip()).casefold() for i in edges
                if not lines[i].lstrip().startswith("#")}
        counts.update(keys)
    threshold = max(2, (len(pages) + 1) // 2)
    cleaned = []
    for lines, edges in zip(page_lines, edge_lines):
        kept = []
        for i, line in enumerate(lines):
            key = re.sub(r"\d+", "#", line.strip()).casefold()
            if i in edges and not line.lstrip().startswith("#") and counts[key] >= threshold:
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))
    return "\n\n".join(cleaned)


def _section_chunks(section, encoder, size, overlap):
    """Choose topic breaks and natural boundaries before enforcing a hard limit."""
    paragraphs = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", section, re.DOTALL))
    breaks = set()
    if encoder is not None and len(paragraphs) > 1:
        vectors = encoder.encode([m.group()[:size] for m in paragraphs], normalize_embeddings=True)
        for i in range(1, len(paragraphs)):
            similarity = sum(float(a) * float(b) for a, b in zip(vectors[i - 1], vectors[i]))
            if similarity < 0.5:
                breaks.add(paragraphs[i].start())
    offset = 0
    while offset < len(section):
        end = min(offset + size, len(section))
        semantic = [point for point in breaks if offset + overlap < point <= end]
        if semantic:
            end = min(semantic)
        elif end < len(section):
            start_search = offset + size // 2
            candidates = [section.rfind(mark, start_search, end) for mark in ("\n\n", "。", ". ", "! ", "? ")]
            boundary = max(candidates)
            if boundary > offset:
                end = boundary + 1
        yield offset, section[offset:end].strip()
        if end >= len(section):
            break
        offset = max(offset + 1, end - overlap)


def chunks_for_paper(paper, size=None, overlap=None, embedding_model=None):
    if size is None:
        size = int(os.environ.get("PAPER_CHUNK_SIZE", "1024"))
    if overlap is None:
        overlap = int(os.environ.get("PAPER_CHUNK_OVERLAP", "50"))
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("overlap must be nonnegative and smaller than size")
    text = remove_page_margins(paper.get("text", "")).strip()
    if not text or not paper.get("title"):
        raise ValueError("each paper requires title and text")
    metadata = {key: paper.get(key, "") for key in ("title", "authors", "source", "doi")}
    identity = paper.get("id") or hashlib.sha256(
        (str(metadata["doi"]) + text).encode("utf-8")
    ).hexdigest()
    sections = re.split(r"(?m)^(#{1,6})[ \t]+(.+?)\s*$", text)
    headings = []
    result = []
    for i in range(0, len(sections), 3):
        if i:
            level = len(sections[i - 2])
            headings = headings[:level - 1]
            headings.append(sections[i - 1].strip())
        heading = " > ".join(headings)
        section = sections[i].strip()
        for offset, body in _section_chunks(section, embedding_model, size, overlap):
            if body:
                chunk_id = hashlib.sha256(f"{identity}:{heading}:{offset}:{body}".encode()).hexdigest()[:20]
                result.append({"id": chunk_id, "paper_id": identity, "section": heading, "text": body, **metadata})
    return result


def build_corpus(source, destination, embedding_model=None):
    source = Path(source)
    papers = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(papers, list):
        raise TypeError("corpus input must be a JSON array")
    if embedding_model is None:
        embedding_model = load_embedding_model()
    chunks = [chunk for paper in papers for chunk in chunks_for_paper(paper, embedding_model=embedding_model)]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return len(chunks)
