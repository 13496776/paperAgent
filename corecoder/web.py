"""FastAPI interface for authenticated research workflows."""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import jwt
import redis
from fastapi import Depends, FastAPI, HTTPException
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
def get_tools():
    index = os.environ.get("PAPER_CORPUS_PATH")
    if not index:
        return ResearchTools()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from .papers.retrieval import Retriever
    chunks = json.loads(Path(index).read_text(encoding="utf-8"))
    embeddings = SentenceTransformer(os.environ.get("PAPER_EMBED_MODEL", "BAAI/bge-m3"))
    reranker = CrossEncoder(os.environ.get("PAPER_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    return ResearchTools(Retriever(chunks, embeddings, reranker))


def supervisor(provider):
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
    return Supervisor(LLM(model=model, api_key=key, base_url=base), get_store(), get_tools())


def handle(call):
    try:
        return call()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sessions/{session}")
def get_session(session: str, user: str = Depends(current_user)):
    return handle(lambda: get_store().load(user, session))


@app.post("/api/sessions/{session}/plan")
def plan(session: str, request: PlanRequest, user: str = Depends(current_user)):
    return handle(lambda: supervisor(request.provider).start(user, session, request.question))


@app.post("/api/sessions/{session}/resume")
def resume(session: str, request: ResumeRequest, user: str = Depends(current_user)):
    return handle(lambda: supervisor(request.provider).resume(user, session))


@app.post("/api/sessions/{session}/cancel")
def cancel(session: str, user: str = Depends(current_user)):
    return handle(lambda: Supervisor(None, get_store()).cancel(user, session))
