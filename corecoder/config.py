"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    """Load .env from cwd, walking up to home dir."""
    from dotenv import load_dotenv

    # search cwd first, then parent dirs up to ~
    env_path = Path(".env")
    if not env_path.exists():
        cur = Path.cwd()
        home = Path.home()
        while cur != home and cur != cur.parent:
            candidate = cur / ".env"
            if candidate.exists():
                env_path = candidate
                break
            cur = cur.parent
    load_dotenv(env_path, override=False)


@dataclass
class Config:
    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"

    @classmethod
    def from_env(cls, prefer_deepseek: bool = True) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        core_key = os.getenv("CORECODER_API_KEY")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        provider = os.getenv("CORECODER_PROVIDER", "openai")
        use_deepseek = bool(prefer_deepseek and deepseek_key and not core_key and provider != "litellm")
        api_key = core_key or (deepseek_key if use_deepseek else os.getenv("OPENAI_API_KEY")) or deepseek_key or ""
        if use_deepseek:
            default_model = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
            base_url = (os.getenv("CORECODER_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
                        or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com")
        else:
            default_model = "gpt-5.5"
            base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("CORECODER_BASE_URL")
                        or (os.getenv("DEEPSEEK_BASE_URL") if not os.getenv("OPENAI_API_KEY") else None))
        return cls(
            model=os.getenv("CORECODER_MODEL") or default_model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=int(os.getenv("CORECODER_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("CORECODER_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("CORECODER_MAX_CONTEXT", "128000")),
            provider=provider,
        )
