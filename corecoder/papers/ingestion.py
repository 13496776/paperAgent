"""Import user-provided and arXiv papers into per-user SQLite corpora."""
import hashlib
import io
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
from pypdf import PdfReader

from .storage import CorpusStore  # noqa: F401

MAX_PDF_BYTES = 20 * 1024 * 1024
_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?\Z")


def _retry_arxiv(operation):
    """Retry read-only arXiv requests, including failures while streaming the PDF."""
    for attempt in range(3):
        try:
            return operation()
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
            response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
            if response is not None and response.status_code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            delay = 2 ** attempt + random.uniform(0, 1)
            value = response.headers.get("Retry-After") if response is not None else None
            if value:
                try:
                    retry_after = float(value)
                except ValueError:
                    try:
                        retry_after = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                    except (TypeError, ValueError, OverflowError):
                        retry_after = 0
                if math.isfinite(retry_after):
                    if retry_after > 60:
                        raise
                    delay = max(delay, retry_after)
            time.sleep(delay)


def extract_pdf(data, filename, *, title="", authors="", source="", doi="", ccf_rank=None,
                venue="", year=None, month=None, date_source="", ccf_source="", ccf_version="", ccf_verified=False):
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
    text = "\f".join(pages).strip(" \t\r\n")
    if not any(page.strip() for page in pages):
        raise ValueError("PDF contains no extractable text; scanned PDFs require OCR")
    # References often contain other papers' DOIs: only inspect metadata and the first page.
    doi_text = " ".join(str(info.get(key) or "") for key in ("/DOI", "/Subject", "/Keywords")) + " " + pages[0]
    detected_doi = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi_text, re.IGNORECASE)
    return {"id": hashlib.sha256(data).hexdigest(), "title": title.strip() or str(info.get("/Title") or "").strip()
            or Path(filename).stem, "authors": authors.strip() or str(info.get("/Author") or "").strip(),
            "source": source.strip() or filename, "doi": doi.strip() or (detected_doi.group().rstrip(".,;") if detected_doi else ""),
            "page_count": len(pages), "ccf_rank": ccf_rank or None, "venue": venue.strip(), "year": year,
            "month": month, "date_source": date_source or ("user" if year is not None else ""),
            "ccf_source": ccf_source.strip(), "ccf_version": ccf_version.strip(), "ccf_verified": ccf_verified, "text": text}



def search_arxiv(query, client=None):
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("query must contain 1–500 characters")
    request = client or httpx
    def fetch():
        response = request.get("https://export.arxiv.org/api/query", params={"search_query": "all:" + query,
                               "start": 0, "max_results": 10}, timeout=20, follow_redirects=False)
        response.raise_for_status()
        return response
    response = _retry_arxiv(fetch)
    return parse_arxiv_feed(response.content)


def parse_arxiv_feed(content):
    root = ET.fromstring(content)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        ident = entry.findtext("a:id", "", ns).rstrip("/").split("/")[-1]
        if not _ARXIV_ID.fullmatch(ident):
            continue
        published = entry.findtext("a:published", "", ns)
        date = re.match(r"^([1-9]\d{3})-(0[1-9]|1[0-2])(?:-\d{2})?(?:T.*)?$", published)
        results.append({"id": ident, "title": " ".join(entry.findtext("a:title", "", ns).split()),
                        "authors": [author.findtext("a:name", "", ns) for author in entry.findall("a:author", ns)],
                        "abstract": " ".join(entry.findtext("a:summary", "", ns).split()),
                        "doi": entry.findtext("arxiv:doi", "", ns).strip(), "source": "https://arxiv.org/abs/" + ident,
                        "year": int(date.group(1)) if date else None, "month": int(date.group(2)) if date else None,
                        "date_source": "arxiv_published" if date else ""})
    return results


def download_arxiv(ident, client=None):
    if not isinstance(ident, str) or not _ARXIV_ID.fullmatch(ident):
        raise ValueError("invalid arXiv identifier")
    request = client or httpx
    # Stream with a fixed host and a hard size bound, without following redirects.
    def fetch():
        with request.stream("GET", "https://arxiv.org/pdf/" + ident, timeout=60, follow_redirects=False) as response:
            response.raise_for_status()
            data = bytearray()
            for part in response.iter_bytes():
                data.extend(part)
                if len(data) > MAX_PDF_BYTES:
                    raise ValueError("PDF exceeds 20 MiB")
        return bytes(data)
    return _retry_arxiv(fetch)


def get_arxiv_paper(ident, client=None):
    if not isinstance(ident, str) or not _ARXIV_ID.fullmatch(ident):
        raise ValueError("invalid arXiv identifier")
    request = client or httpx
    def fetch():
        response = request.get("https://export.arxiv.org/api/query", params={"id_list": ident}, timeout=20, follow_redirects=False)
        response.raise_for_status()
        return response
    response = _retry_arxiv(fetch)
    papers = parse_arxiv_feed(response.content)
    if not papers or papers[0]["id"].split("v")[0] != ident.split("v")[0]:
        raise ValueError("arXiv paper not found")
    paper = papers[0]
    data = download_arxiv(paper["id"], client=client)
    return extract_pdf(data, paper["id"] + ".pdf", title=paper["title"], authors=", ".join(paper["authors"]),
                       source=paper["source"], doi=paper["doi"], year=paper["year"], month=paper["month"],
                       date_source=paper["date_source"])
