"""Import user-provided and arXiv papers into per-user SQLite corpora."""
import hashlib
import io
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from pypdf import PdfReader

from .corpus import chunks_for_paper

MAX_PDF_BYTES = 20 * 1024 * 1024
_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?\Z")


def extract_pdf(data, filename, *, title="", authors="", source="", doi=""):
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds 20 MiB")
    if not data.startswith(b"%PDF-"):
        raise ValueError("file is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("encrypted PDF cannot be opened")
        pages = [page.extract_text() or "" for page in reader.pages]
        info = reader.metadata or {}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF could not be parsed") from exc
    text = "\f".join(pages).strip()
    if not text:
        raise ValueError("PDF contains no extractable text; scanned PDFs require OCR")
    detected_doi = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
    return {"id": hashlib.sha256(data).hexdigest(), "title": title.strip() or str(info.get("/Title") or "").strip()
            or Path(filename).stem, "authors": authors.strip() or str(info.get("/Author") or "").strip(),
            "source": source.strip() or filename, "doi": doi.strip() or (detected_doi.group().rstrip(".,;") if detected_doi else ""),
            "text": text}


class CorpusStore:
    def __init__(self, path):
        self.path = str(path)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS paper_corpus (owner TEXT NOT NULL, id TEXT NOT NULL, "
                       "metadata TEXT NOT NULL, chunks TEXT NOT NULL, PRIMARY KEY (owner, id))")

    def add(self, owner, paper, encoder):
        chunks = chunks_for_paper(paper, embedding_model=encoder)
        if not chunks:
            raise ValueError("paper contains no searchable text")
        metadata = {key: paper[key] for key in ("id", "title", "authors", "source", "doi")}
        with sqlite3.connect(self.path, timeout=30) as db:
            cursor = db.execute("INSERT OR IGNORE INTO paper_corpus VALUES (?, ?, ?, ?)",
                                (owner, paper["id"], json.dumps(metadata, ensure_ascii=False),
                                 json.dumps(chunks, ensure_ascii=False)))
            created = cursor.rowcount == 1
        return {**metadata, "chunks": len(chunks), "created": created}

    def papers(self, owner):
        with sqlite3.connect(self.path) as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT metadata FROM paper_corpus WHERE owner=? ORDER BY rowid DESC", (owner,))]

    def chunks(self, owner):
        with sqlite3.connect(self.path) as db:
            return [chunk for row in db.execute("SELECT chunks FROM paper_corpus WHERE owner=?", (owner,))
                    for chunk in json.loads(row[0])]


def search_arxiv(query, client=None):
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("query must contain 1–500 characters")
    request = client or httpx
    response = request.get("https://export.arxiv.org/api/query", params={"search_query": "all:" + query,
                           "start": 0, "max_results": 10}, timeout=20, follow_redirects=False)
    response.raise_for_status()
    return parse_arxiv_feed(response.content)


def parse_arxiv_feed(content):
    root = ET.fromstring(content)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        ident = entry.findtext("a:id", "", ns).rstrip("/").split("/")[-1]
        if not _ARXIV_ID.fullmatch(ident):
            continue
        results.append({"id": ident, "title": " ".join(entry.findtext("a:title", "", ns).split()),
                        "authors": [author.findtext("a:name", "", ns) for author in entry.findall("a:author", ns)],
                        "abstract": " ".join(entry.findtext("a:summary", "", ns).split()),
                        "doi": entry.findtext("arxiv:doi", "", ns), "source": "https://arxiv.org/abs/" + ident})
    return results


def download_arxiv(ident, client=None):
    if not isinstance(ident, str) or not _ARXIV_ID.fullmatch(ident):
        raise ValueError("invalid arXiv identifier")
    request = client or httpx
    # Stream with a fixed host and a hard size bound, without following redirects.
    with request.stream("GET", "https://arxiv.org/pdf/" + ident, timeout=60, follow_redirects=False) as response:
        response.raise_for_status()
        data = bytearray()
        for part in response.iter_bytes():
            data.extend(part)
            if len(data) > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds 20 MiB")
    return bytes(data)


def get_arxiv_paper(ident, client=None):
    if not isinstance(ident, str) or not _ARXIV_ID.fullmatch(ident):
        raise ValueError("invalid arXiv identifier")
    request = client or httpx
    response = request.get("https://export.arxiv.org/api/query", params={"id_list": ident}, timeout=20, follow_redirects=False)
    response.raise_for_status()
    papers = parse_arxiv_feed(response.content)
    if not papers or papers[0]["id"].split("v")[0] != ident.split("v")[0]:
        raise ValueError("arXiv paper not found")
    paper = papers[0]
    data = download_arxiv(paper["id"], client=client)
    return extract_pdf(data, paper["id"] + ".pdf", title=paper["title"], authors=", ".join(paper["authors"]),
                       source=paper["source"], doi=paper["doi"])
