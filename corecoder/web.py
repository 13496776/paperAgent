"""FastAPI interface for authenticated research workflows."""
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import httpx
import jwt
import redis
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import BaseModel, Field

from .llm import LLM
from .persistent import JWTAuthenticator, SessionStore
from .runtime_config import apply_runtime_config
from .workflow.engine import ResearchTools, Supervisor, error_message

apply_runtime_config()
app = FastAPI(title="Paper Research Agent")
security = HTTPBearer()
_bootstrap_lock = threading.Lock()


def default_provider() -> Literal["openai", "deepseek"]:
    if os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
        return "deepseek"
    return "openai"


class PlanRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    provider: Literal["openai", "deepseek"] = Field(default_factory=default_provider)


class ResumeRequest(BaseModel):
    provider: Literal["openai", "deepseek"] = Field(default_factory=default_provider)


def current_user(credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]):
    try:
        auth = JWTAuthenticator(os.environ["CORECODER_JWT_KEY"], os.environ["CORECODER_JWT_ISSUER"],
                                os.environ["CORECODER_JWT_AUDIENCE"])
        return auth.authenticate(credentials.credentials)
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
    except KeyError as exc:
        raise HTTPException(status_code=503, detail="Authentication is not configured") from exc


@lru_cache(maxsize=1)
def get_store():
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is not configured")
    return SessionStore(redis.Redis.from_url(url), os.environ.get("CORECODER_SQLITE_PATH", "corecoder-memory.sqlite3"))


@lru_cache(maxsize=1)
def get_models():
    from sentence_transformers import CrossEncoder

    from .papers.corpus import load_embedding_model

    return load_embedding_model(), CrossEncoder(os.environ.get("PAPER_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))


def corpus_store():
    from .papers.storage import CorpusStore

    return CorpusStore(os.environ.get("CORECODER_SQLITE_PATH", "corecoder-memory.sqlite3"))


def import_arxiv(user, ident):
    from .papers.ingestion import get_arxiv_paper

    return corpus_store().add(user, get_arxiv_paper(ident), get_models()[0])


def import_paperlist(user, ident, *, record=None):
    from .papers.paperlist import _normalize_paper, get_paperlist

    record = _normalize_paper(record) if record is not None else get_paperlist(ident)
    if record["id"] != ident:
        raise ValueError("Paperlist 论文 ID 不匹配")
    conference = record.get("conference") if isinstance(record.get("conference"), dict) else {}
    venue = conference.get("name") if isinstance(conference.get("name"), str) else ""
    year = conference.get("year")
    if type(year) is not int or not 1000 <= year <= 9999:
        year = None
    month = None
    date_source = "paperlist_conference_year" if year else ""
    published = record.get("published_at")
    if isinstance(published, str):
        try:
            published_date = date.fromisoformat(published[:10])
            year, month, date_source = published_date.year, published_date.month, "paperlist_published_at"
        except ValueError:
            pass
    paper = {"id": hashlib.sha256(("paperlist:" + ident).encode("utf-8")).hexdigest(),
             "provider": "paperlist", "external_id": ident, "title": record["title"], "authors": record["authors"],
             "source": record["url"], "doi": record["doi"], "venue": venue,
             "year": year, "month": month, "date_source": date_source, "page_count": None}
    abstract = record["abstract"].strip()
    store = corpus_store()
    metadata = store.add_metadata(user, paper)
    if not abstract or (metadata["content_scope"] == "abstract" and metadata["status"] == "ready"):
        return metadata
    paper.update(text=abstract, content_scope="abstract")
    try:
        result = store.add(user, paper, get_models()[0])
    except Exception as exc:
        raise RuntimeError(f"Paperlist 题录已保存，但摘要索引失败：{error_message(exc)}") from exc
    result["created"] = metadata["created"] or result["created"]
    return result


_shared_index_lock = threading.Lock()


def _paper_matches(query, papers):
    """Only unambiguous exact identifiers, full titles, or title-prefix aliases."""
    text = unicodedata.normalize('NFKC', query).casefold()
    strong, aliases = [], []

    def mentioned(value):
        return bool(re.search(r'(?<![a-z0-9])' + re.escape(value.casefold()) + r'(?![a-z0-9])', text))

    for paper in papers:
        title = unicodedata.normalize('NFKC', paper['title']).strip()
        prefix = title.split(':', 1)[0].strip() if ':' in title else ''
        ident = str(paper['id'])
        if (len(title) >= 8 and mentioned(title)) or (len(ident) >= 16 and mentioned(ident)):
            strong.append((paper, prefix.casefold()))
        elif 3 <= len(prefix) <= 50 and mentioned(prefix):
            aliases.append((paper, prefix.casefold()))
    # An exact full title disambiguates competing papers sharing its prefix,
    # but an independent second alias still makes the query multi-paper.
    disambiguated = {prefix for _, prefix in strong if prefix}
    return [paper for paper, _ in strong] + [paper for paper, prefix in aliases if prefix not in disambiguated]


class UserRetriever:
    def __init__(self, user):
        self.user = user

    def list_papers(self, year=None):
        store = corpus_store()
        papers = store.papers(self.user, year=year)
        known = {paper["id"] for paper in papers}
        unknown_year_count = 0
        if year is not None:
            with store._connect() as db:
                unknown_year_count = db.execute("SELECT COUNT(*) FROM papers WHERE owner=? AND status!='deleting' AND year IS NULL", (self.user,)).fetchone()[0]
        index = os.environ.get("PAPER_CORPUS_PATH")
        if index:
            if year is not None:
                with store._connect() as db:
                    known.update(row[0] for row in db.execute("SELECT id FROM papers WHERE owner=?", (self.user,)))
            chunks = json.loads(Path(index).read_text(encoding="utf-8"))
            for chunk in chunks:
                ident = chunk.get("paper_id")
                if ident and ident not in known:
                    known.add(ident)
                    if year is not None and chunk.get("year") != year:
                        if chunk.get("year") is None:
                            unknown_year_count += 1
                        continue
                    papers.append({"id": ident, "title": chunk.get("title", ""), "authors": chunk.get("authors", ""),
                                   "source": chunk.get("source", ""), "doi": chunk.get("doi", ""),
                                   "year": chunk.get("year"), "date_source": chunk.get("date_source", "")})
        result = {"count": len(papers), "papers": papers}
        if year is not None:
            result.update(year=year, unknown_year_count=unknown_year_count)
        return result

    def retrieve(self, query, paper_id=None):
        return self.retrieve_with_progress(query, paper_id=paper_id)['hits']

    def retrieve_with_progress(self, query, progress=None, paper_id=None):
        from .papers.retrieval import bm25, fuse_and_rerank

        limit = int(os.environ.get("PAPER_RETRIEVAL_LIMIT", "5"))
        candidates = int(os.environ.get("PAPER_RETRIEVAL_CANDIDATES", "30"))
        threshold = float(os.environ.get("PAPER_RERANK_THRESHOLD", "0.3"))
        if limit < 1 or candidates < 1 or not 0 <= threshold <= 1:
            raise ValueError("retrieval limits must be positive and rerank threshold must be between 0 and 1")
        if progress:
            progress("corpus_load", "读取论文元数据")
        store = corpus_store()
        papers = store.papers(self.user)
        if progress:
            progress('index_inventory', '检查论文的索引状态与首个分块向量')
        states = store.index_states(self.user)
        matches = _paper_matches(query, papers)
        if paper_id is not None:
            if not any(paper['id'] == paper_id for paper in papers):
                raise ValueError('指定论文不存在或不属于当前用户')
            if len(matches) == 1 and matches[0]['id'] != paper_id:
                raise ValueError('指定论文 ID 与查询中的论文名称不一致')
            target = paper_id
        else:
            target = matches[0]['id'] if len(matches) == 1 else None
        index = os.environ.get("PAPER_CORPUS_PATH")
        shared = json.loads(Path(index).read_text(encoding="utf-8")) if index and paper_id is None else []
        selected = next((s for s in states if s['id'] == target), None)
        if selected and selected['vector_ready'] and progress:
            progress('index_verify', f"核对指定论文的分块向量：{selected['title']}（ID: {target}）")
        missing_vectors = bool(selected and selected['vector_ready'] and not store.paper_vectors_complete(self.user, target))
        if selected and selected['content_scope'] != 'metadata_only' and (not selected['vector_ready'] or missing_vectors):
            if progress:
                progress('index_target', f"仅构建或恢复指定论文索引：{selected['title']}（ID: {target}）")
            embedding, reranker = get_models()
            # A missing collection/vector can coexist with a stale 'ready'
            # SQLite row; rebuild this one paper rather than trusting it.
            force = selected['status'] == 'ready' and (not selected['vector_ready'] or missing_vectors)
            store.ensure_indexed(self.user, embedding, paper_id=target, progress=progress, force=force)
            states = store.index_states(self.user)
        scoped = [s for s in states if paper_id is None or s['id'] == paper_id]
        pending = [s for s in scoped if s['content_scope'] != 'metadata_only' and not s['vector_ready']]
        metadata_only = [s for s in scoped if s['content_scope'] == 'metadata_only']
        coverage = {'scope': 'paper' if paper_id else 'corpus', 'paper_id': paper_id,
                    'matched_paper_id': target, 'ambiguous_match_count': len(matches) if len(matches) > 1 else 0,
                    'vector_check': '按论文检查元数据和首个分块向量；单篇目标会逐块核对。',
                    'vector_ready_count': sum(s['vector_ready'] for s in scoped),
                    'pending_count': len(pending), 'metadata_only_count': len(metadata_only),
                    'pending_papers': [{'id': s['id'], 'title': s['title'], 'status': s['status']} for s in pending[:10]],
                    'shared_vector_ready': None, 'warnings': []}
        if pending:
            coverage['warnings'].append(f"{len(pending)} 篇论文的向量索引未就绪；本次不会自动重建全库，搜索可能不完整。")
        if metadata_only:
            coverage['warnings'].append(f"{len(metadata_only)} 篇论文只有元数据，无法检索正文或摘要。")
        if len(matches) > 1 and paper_id is None:
            coverage['warnings'].append('查询匹配多篇论文，未自动选择目标或构建索引；请提供唯一论文 ID。')
        if progress and coverage['warnings']:
            progress('index_coverage', '；'.join(coverage['warnings']))
        if not any(s['status'] == 'ready' for s in scoped) and not shared:
            return {'hits': [], 'coverage': coverage}
        if progress:
            progress("model_loading", "加载或复用嵌入模型和重排模型")
        embedding, reranker = get_models()
        if progress:
            progress("query_embedding", "编码查询向量")
        vector = embedding.encode([query], normalize_embeddings=True)[0]
        if progress:
            progress("bm25", "SQLite 关键词检索")
        lexical = store.lexical_search(self.user, query, candidates, paper_id=paper_id)
        if progress:
            progress("vector_search", "Chroma 向量检索")
        dense = store.vector_search(self.user, vector, candidates, paper_id=paper_id)
        keys = list(dict.fromkeys(lexical + dense))
        personal = store.chunks_for_keys(self.user, keys)
        ranked_lists = [[("personal", *item) for item in lexical], [("personal", *item) for item in dense]]
        chunks = {("personal", *key): {**chunk, "_scope": "personal"} for key, chunk in personal.items()}
        # Shared JSON corpora retain their file-backed metadata and use a persistent Chroma index.
        if shared:
            if progress:
                progress("bm25_shared", "共享语料关键词检索")
            scores = bm25(query, shared)
            shared_lex = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidates]
            if progress:
                progress("index_wait", "检查或等待共享语料向量索引")
            with _shared_index_lock:
                shared_dense = store.shared_vector_search(shared, vector, embedding, candidates, progress, build=False)
            coverage['shared_vector_ready'] = shared_dense is not None
            if shared_dense is None:
                coverage['warnings'].append('共享语料向量索引未就绪，本次仅使用其关键词检索。')
                if progress:
                    progress('index_coverage', coverage['warnings'][-1])
                shared_dense = []
            ranked_lists.extend([[('shared', i) for i in shared_lex], [('shared', i) for i in shared_dense]])
            chunks.update({('shared', i): {**shared[i], "_scope": "shared"} for i in set(shared_lex + shared_dense)})
        results = fuse_and_rerank(query, ranked_lists, chunks, reranker, limit, threshold, progress)
        personal_keys = [(chunk["paper_id"], chunk["id"]) for chunk in results if chunk.get("_scope") == "personal"]
        still_available = store.chunks_for_keys(self.user, personal_keys)
        hits = [{key: value for key, value in chunk.items() if key != "_scope"} for chunk in results
                if chunk.get("_scope") != "personal" or (chunk["paper_id"], chunk["id"]) in still_available]
        return {'hits': hits, 'coverage': coverage}


def get_tools(user):
    from .papers.ingestion import search_arxiv
    from .papers.paperlist import search_paperlist

    if os.environ.get("PAPER_EVALUATION_LOCAL_ONLY") == "1":
        return ResearchTools(UserRetriever(user))
    return ResearchTools(UserRetriever(user), online_search=search_arxiv,
                         online_import=lambda ident: import_arxiv(user, ident),
                         paperlist_search=search_paperlist,
                         paperlist_import=lambda ident, record=None: import_paperlist(user, ident, record=record))


def supervisor(provider, user):
    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    else:
        key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        base = os.environ.get("OPENAI_BASE_URL")
    if not key:
        raise RuntimeError(f"{provider} API key is not configured")
    timeout = float(os.environ.get("PAPER_LLM_TIMEOUT", "60"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("llm_timeout_seconds must be positive and finite")
    return Supervisor(LLM(model=model, api_key=key, base_url=base, timeout=timeout, client_max_retries=0),
                      get_store(), get_tools(user))


def handle(call):
    try:
        return call()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=502, detail="模型服务认证失败，请检查所选提供商及 API Key") from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=502, detail="模型服务限流或额度不足，请检查账户状态") from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise HTTPException(status_code=502, detail="无法连接模型服务或请求超时，请检查网络和服务地址") from exc
    except APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"模型服务返回 HTTP {exc.status_code}，请检查模型配置") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=error_message(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class LocalBootstrapRequest(BaseModel):
    code: str = Field(min_length=1, max_length=256)


@app.post("/api/local-session")
def local_session(request: LocalBootstrapRequest, response: Response):
    # The launcher supplies a high-entropy, short-lived, one-use code only to
    # the loopback-bound development server and the browser URL fragment.
    with _bootstrap_lock:
        expected = os.environ.get("PAPER_LOCAL_BOOTSTRAP_CODE")
        expires = os.environ.get("PAPER_LOCAL_BOOTSTRAP_EXPIRES", "0")
        try:
            valid = expected and hmac.compare_digest(request.code.encode("utf-8"), expected.encode("utf-8")) and time.time() < float(expires)
        except ValueError:
            valid = False
        if not valid:
            raise HTTPException(status_code=401, detail="Local sign-in link is invalid or expired")
        now = int(time.time())
        token = jwt.encode({"sub": "local-user", "iss": os.environ["CORECODER_JWT_ISSUER"],
                            "aud": os.environ["CORECODER_JWT_AUDIENCE"], "iat": now, "exp": now + 86400},
                           os.environ["CORECODER_JWT_KEY"], algorithm="HS256")
        os.environ.pop("PAPER_LOCAL_BOOTSTRAP_CODE", None)
        os.environ.pop("PAPER_LOCAL_BOOTSTRAP_EXPIRES", None)
    response.headers["Cache-Control"] = "no-store"
    return {"token": token}


@app.get("/api/settings")
def public_settings():
    return {"default_provider": default_provider(),
            "evaluation_local_only": os.environ.get("PAPER_EVALUATION_LOCAL_ONLY") == "1",
            "paperlist_enabled": bool(os.environ.get("PAPERLIST_API_KEY")),
            "shared_corpus_enabled": bool(os.environ.get("PAPER_CORPUS_PATH"))}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sessions/{session}")
def get_session(session: str, user: str = Depends(current_user)):
    return handle(lambda: Supervisor(None, get_store()).get(user, session))


@app.post("/api/sessions/{session}/plan")
def plan(session: str, request: PlanRequest, user: str = Depends(current_user)):
    return handle(lambda: supervisor(request.provider, user).start(user, session, request.question))


@app.post("/api/sessions/{session}/resume")
def resume(session: str, request: ResumeRequest, user: str = Depends(current_user)):
    return handle(lambda: supervisor(request.provider, user).resume(user, session))


@app.post("/api/sessions/{session}/pause")
def pause(session: str, user: str = Depends(current_user)):
    return handle(lambda: Supervisor(None, get_store()).pause(user, session))


@app.post("/api/sessions/{session}/cancel")
def cancel(session: str, user: str = Depends(current_user)):
    return handle(lambda: Supervisor(None, get_store()).cancel(user, session))


class ArxivSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class PaperlistSearchRequest(ArxivSearchRequest):
    query: str = Field(min_length=1, max_length=300)
    cursor: str | None = Field(default=None, min_length=1, max_length=2000)


class ArxivImportRequest(BaseModel):
    id: str = Field(min_length=1, max_length=32)


class PaperlistImportRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)


@app.get("/api/corpus")
def list_papers(user: str = Depends(current_user)):
    return handle(lambda: corpus_store().papers(user))


@app.post("/api/corpus/upload")
def upload_paper(file: Annotated[UploadFile, File()], title: str = Form(""), authors: str = Form(""),
                 doi: str = Form(""), source: str = Form(""), ccf_rank: str = Form(""), venue: str = Form(""),
                 year: int | None = Form(None), month: int | None = Form(None), ccf_source: str = Form(""), ccf_version: str = Form(""),
                 ccf_verified: bool = Form(False), user: str = Depends(current_user)):
    from .papers.ingestion import MAX_PDF_BYTES, extract_pdf

    def ingest():
        data = file.file.read(MAX_PDF_BYTES + 1)
        paper = extract_pdf(data, file.filename or "paper.pdf", title=title, authors=authors, doi=doi,
                            source=source, ccf_rank=ccf_rank or None, venue=venue, year=year, month=month,
                            ccf_source=ccf_source, ccf_version=ccf_version, ccf_verified=ccf_verified)
        return corpus_store().add(user, paper, get_models()[0])
    return handle(ingest)


@app.post("/api/corpus/search")
def search_papers(request: ArxivSearchRequest, user: str = Depends(current_user)):
    from .papers.ingestion import search_arxiv

    return handle(lambda: search_arxiv(request.query))


@app.post("/api/corpus/paperlist/search")
def search_paperlist_papers(request: PaperlistSearchRequest, user: str = Depends(current_user)):
    from .papers.paperlist import search_paperlist

    return handle(lambda: search_paperlist(request.query, cursor=request.cursor))


@app.post("/api/corpus/import")
def import_paper(request: ArxivImportRequest, user: str = Depends(current_user)):
    return handle(lambda: import_arxiv(user, request.id))


@app.post("/api/corpus/paperlist/import")
def import_paperlist_paper(request: PaperlistImportRequest, user: str = Depends(current_user)):
    return handle(lambda: import_paperlist(user, request.id))


@app.delete("/api/corpus/{paper_id}")
def delete_paper(paper_id: str, user: str = Depends(current_user)):
    def remove():
        if not corpus_store().delete(user, paper_id):
            raise HTTPException(status_code=404, detail="论文不存在")
        return {"deleted": True}
    return handle(remove)
