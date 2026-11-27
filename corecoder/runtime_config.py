"""Optional local JSON configuration shared by CLI and web entry points."""
import json
import os
from pathlib import Path

FIELDS = {
    "embedding_model": "PAPER_EMBED_MODEL",
    "reranker_model": "PAPER_RERANK_MODEL",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "paperlist_api_key": "PAPERLIST_API_KEY",
    "deepseek_model": "DEEPSEEK_MODEL",
    "openai_model": "OPENAI_MODEL",
    "deepseek_base_url": "DEEPSEEK_BASE_URL",
    "openai_base_url": "OPENAI_BASE_URL",
    "redis_url": "REDIS_URL",
    "sqlite_path": "CORECODER_SQLITE_PATH",
    "chroma_path": "PAPER_CHROMA_PATH",
    "chunk_size": "PAPER_CHUNK_SIZE",
    "chunk_overlap": "PAPER_CHUNK_OVERLAP",
    "rerank_threshold": "PAPER_RERANK_THRESHOLD",
    "retrieval_limit": "PAPER_RETRIEVAL_LIMIT",
    "retrieval_candidates": "PAPER_RETRIEVAL_CANDIDATES",
    "retrieval_timeout_seconds": "PAPER_RETRIEVAL_TIMEOUT",
    "retrieval_lock_wait_seconds": "PAPER_RETRIEVAL_LOCK_WAIT",
    "llm_timeout_seconds": "PAPER_LLM_TIMEOUT",
    "arxiv_search_interval_seconds": "PAPER_ARXIV_SEARCH_INTERVAL",
    "arxiv_api_timeout_seconds": "PAPER_ARXIV_API_TIMEOUT",
}


def apply_runtime_config(path=None):
    """Apply nonempty JSON values unless an environment variable is already set."""
    if path is None and os.environ.get("PAPER_DISABLE_CONFIG") == "1":
        return
    config_path = Path(path or os.environ.get("PAPER_CONFIG_PATH", "paperagent.config.json"))
    if not config_path.is_file():
        if path or os.environ.get("PAPER_CONFIG_PATH"):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise TypeError("Configuration must be a JSON object")
    unknown = set(data) - FIELDS.keys()
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    for field, value in data.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError(f"Invalid value for {field}")
        os.environ.setdefault(FIELDS[field], str(value))
