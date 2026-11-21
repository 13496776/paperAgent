"""FastAPI interface for authenticated research workflows."""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import httpx
import jwt
import redis
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .llm import LLM
from .persistent import JWTAuthenticator, SessionStore
from .workflow.engine import ResearchTools, Supervisor

app = FastAPI(title="Paper Research Agent")
security = HTTPBearer()


class PlanRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    provider: Literal["openai", "deepseek"] = "openai"


class ResumeRequest(BaseModel):
    provider: Literal["openai", "deepseek"] = "openai"


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


class UserRetriever:
    def __init__(self, user):
        self.user = user

    def retrieve(self, query):
        from .papers.retrieval import Retriever

        chunks = corpus_store().chunks(self.user)
        index = os.environ.get("PAPER_CORPUS_PATH")
        if index:
            chunks.extend(json.loads(Path(index).read_text(encoding="utf-8")))
        if not chunks:
            return []
        embedding, reranker = get_models()
        return Retriever(chunks, embedding, reranker).retrieve(query)


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
    return Supervisor(LLM(model=model, api_key=key, base_url=base), get_store(), get_tools(user))


def handle(call):
    try:
        return call()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sessions/{session}")
def get_session(session: str, user: str = Depends(current_user)):
    return handle(lambda: get_store().load(user, session))


@app.post("/api/sessions/{session}/plan")
def plan(session: str, request: PlanRequest, user: str = Depends(current_user)):
    return handle(lambda: supervisor(request.provider, user).start(user, session, request.question))


@app.post("/api/sessions/{session}/resume")
def resume(session: str, request: ResumeRequest, user: str = Depends(current_user)):
    return handle(lambda: supervisor(request.provider, user).resume(user, session))


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
