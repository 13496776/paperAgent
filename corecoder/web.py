"""FastAPI interface for authenticated research workflows."""
import hashlib
import hmac
import json
import math
import os
import threading
import time
from collections import OrderedDict
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
    from .papers.ingestion import CorpusStore

    return CorpusStore(os.environ.get("CORECODER_SQLITE_PATH", "corecoder-memory.sqlite3"))


def import_arxiv(user, ident):
    from .papers.ingestion import get_arxiv_paper

    return corpus_store().add(user, get_arxiv_paper(ident), get_models()[0])


_retriever_cache = OrderedDict()
_retriever_cache_lock = threading.Lock()


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
        from .papers.retrieval import Retriever

        if progress:
            progress("corpus_load", "读取本地论文分块")
        chunks = corpus_store().chunks(self.user)
        index = os.environ.get("PAPER_CORPUS_PATH")
        if index:
            chunks.extend(json.loads(Path(index).read_text(encoding="utf-8")))
        if not chunks:
            return []
        if progress:
            progress("model_loading", "加载或复用嵌入模型和重排模型")
        embedding, reranker = get_models()
        fingerprint = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        key = (self.user, fingerprint, id(embedding), id(reranker))
        if progress:
            progress("index_wait", "检查或等待向量索引")
        with _retriever_cache_lock:
            retriever = _retriever_cache.get(key)
            if retriever is None:
                if progress:
                    progress("index_build", f"构建向量索引，编码 {len(chunks)} 个分块")
                retriever = Retriever(chunks, embedding, reranker)
                for old_key in list(_retriever_cache):
                    if old_key[0] == self.user:
                        del _retriever_cache[old_key]
                _retriever_cache[key] = retriever
                while len(_retriever_cache) > 16:
                    _retriever_cache.popitem(last=False)
            else:
                _retriever_cache.move_to_end(key)
                if progress:
                    progress("index_cache", f"复用已构建的向量索引，包含 {len(chunks)} 个分块")
        return retriever.retrieve(query, progress=progress)


def get_tools(user):
    from .papers.ingestion import search_arxiv

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
    return {"default_provider": default_provider()}


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
                 doi: str = Form(""), source: str = Form(""), user: str = Depends(current_user)):
    from .papers.ingestion import MAX_PDF_BYTES, extract_pdf

    def ingest():
        data = file.file.read(MAX_PDF_BYTES + 1)
        paper = extract_pdf(data, file.filename or "paper.pdf", title=title, authors=authors, doi=doi, source=source)
        return corpus_store().add(user, paper, get_models()[0])
    return handle(ingest)


@app.post("/api/corpus/search")
def search_papers(request: ArxivSearchRequest, user: str = Depends(current_user)):
    from .papers.ingestion import search_arxiv

    return handle(lambda: search_arxiv(request.query))


@app.post("/api/corpus/import")
def import_paper(request: ArxivImportRequest, user: str = Depends(current_user)):
    return handle(lambda: import_arxiv(user, request.id))
