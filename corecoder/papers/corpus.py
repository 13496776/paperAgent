"""Local JSON corpus and structure-aware passage splitting."""
import hashlib
import json
import re
from pathlib import Path


def chunks_for_paper(paper, size=1200, overlap=200):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("overlap must be nonnegative and smaller than size")
    text = paper.get("text", "").strip()
    if not text or not paper.get("title"):
        raise ValueError("each paper requires title and text")
    metadata = {key: paper.get(key, "") for key in ("title", "authors", "source", "doi")}
    identity = paper.get("id") or hashlib.sha256(
        (str(metadata["doi"]) + text).encode("utf-8")
    ).hexdigest()
    # Markdown headings retain section boundaries. Paragraphs and sentences
    # provide natural breakpoints before the hard character limit.
    sections = re.split(r"(?m)^(#{1,6}\s+.+)$", text)
    heading = ""
    result = []
    for section in sections:
        if re.match(r"^#{1,6}\s", section):
            heading = section.lstrip("# ").strip()
            continue
        section = section.strip()
        offset = 0
        while offset < len(section):
            end = min(offset + size, len(section))
            if end < len(section):
                candidates = [section.rfind(mark, offset + size // 2, end) for mark in ("\n\n", "。", ". ", "! ", "? ")]
                boundary = max(candidates)
                if boundary > offset:
                    end = boundary + 1
            body = section[offset:end].strip()
            if body:
                chunk_id = hashlib.sha256(f"{identity}:{heading}:{offset}:{body}".encode()).hexdigest()[:20]
                result.append({"id": chunk_id, "paper_id": identity, "section": heading, "text": body, **metadata})
            if end == len(section):
                break
            offset = max(offset + 1, end - overlap)
    return result


def build_corpus(source, destination):
    source = Path(source)
    papers = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(papers, list):
        raise TypeError("corpus input must be a JSON array")
    chunks = [chunk for paper in papers for chunk in chunks_for_paper(paper)]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return len(chunks)
