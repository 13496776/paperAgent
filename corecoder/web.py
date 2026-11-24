"""FastAPI interface for authenticated research workflows."""
import hmac
import json
import math
import os
import threading
import time
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
from .workflow.engine import ResearchTools, Supervisor

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


_shared_index_lock = threading.Lock()


class UserRetriever:
    def __init__(self, user):
        self.user = user

    def list_papers(self):
        papers = corpus_store().papers(self.user)
        known = {paper["id"] for paper in papers}
        index = os.environ.get("PAPER_CORPUS_PATH")
        if index:
            chunks = json.loads(Path(index).read_text(encoding="utf-8"))
            for chunk in chunks:
                ident = chunk.get("paper_id")
                if ident and ident not in known:
                    papers.append({"id": ident, "title": chunk.get("title", ""), "authors": chunk.get("authors", ""),
                                   "source": chunk.get("source", ""), "doi": chunk.get("doi", "")})
                    known.add(ident)
        return {"count": len(papers), "papers": papers}

    def retrieve(self, query):
        return self.retrieve_with_progress(query)

    def retrieve_with_progress(self, query, progress=None):
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
        index = os.environ.get("PAPER_CORPUS_PATH")
        shared = json.loads(Path(index).read_text(encoding="utf-8")) if index else []
        if not papers and not shared:
            return []
        if progress:
            progress("model_loading", "加载或复用嵌入模型和重排模型")
        embedding, reranker = get_models()
        store.ensure_indexed(self.user, embedding, progress=progress)
        if progress:
            progress("query_embedding", "编码查询向量")
        vector = embedding.encode([query], normalize_embeddings=True)[0]
        if progress:
            progress("bm25", "SQLite 关键词检索")
        lexical = store.lexical_search(self.user, query, candidates)
        if progress:
            progress("vector_search", "Chroma 向量检索")
        dense = store.vector_search(self.user, vector, candidates)
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
                shared_dense = store.shared_vector_search(shared, vector, embedding, candidates, progress)
            ranked_lists.extend([[('shared', i) for i in shared_lex], [('shared', i) for i in shared_dense]])
            chunks.update({('shared', i): {**shared[i], "_scope": "shared"} for i in set(shared_lex + shared_dense)})
        results = fuse_and_rerank(query, ranked_lists, chunks, reranker, limit, threshold, progress)
        personal_keys = [(chunk["paper_id"], chunk["id"]) for chunk in results if chunk.get("_scope") == "personal"]
        still_available = store.chunks_for_keys(self.user, personal_keys)
        return [{key: value for key, value in chunk.items() if key != "_scope"} for chunk in results
                if chunk.get("_scope") != "personal" or (chunk["paper_id"], chunk["id"]) in still_available]


def get_tools(user):
    from .papers.ingestion import search_arxiv

    if os.environ.get("PAPER_EVALUATION_LOCAL_ONLY") == "1":
        return ResearchTools(UserRetriever(user))
    return ResearchTools(UserRetriever(user), online_search=search_arxiv,
                         online_import=lambda ident: import_arxiv(user, ident))


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
    except (RuntimeError, httpx.HTTPError) as exc:
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


class ArxivImportRequest(BaseModel):
    id: str = Field(min_length=1, max_length=32)


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


@app.post("/api/corpus/import")
def import_paper(request: ArxivImportRequest, user: str = Depends(current_user)):
    return handle(lambda: import_arxiv(user, request.id))


@app.delete("/api/corpus/{paper_id}")
def delete_paper(paper_id: str, user: str = Depends(current_user)):
    def remove():
        if not corpus_store().delete(user, paper_id):
            raise HTTPException(status_code=404, detail="论文不存在")
        return {"deleted": True}
    return handle(remove)
