"""Central configuration for the RAG module.

All settings come from environment variables (optionally loaded from
``rag/.env``) with safe defaults. Relative paths are resolved against the
``rag/`` directory, never against the current working directory, so the module
behaves the same whether it is run from ``rag/`` or imported by a backend.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

RAG_ROOT = Path(__file__).resolve().parent.parent

_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


def _load_dotenv() -> None:
    """Load ``rag/.env`` if python-dotenv is installed. Never overrides real env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv(RAG_ROOT / ".env", override=False)


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (RAG_ROOT / path).resolve()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = _env_float_optional(name)
    return default if value is None else value


def _env_float_optional(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class RAGSettings:
    """Immutable settings object. Build with :meth:`from_env` and override with :meth:`with_overrides`."""

    # --- storage ---
    documents_dir: Path
    chroma_dir: Path
    collection_name: str
    # --- project identification ---
    project_folder_level: int  # 0 = disabled; 2 = documents/<type>/<project>/file
    # --- embeddings (local sentence-transformers) ---
    embedding_model: str
    embedding_batch_size: int
    embedding_device: Optional[str]
    embedding_query_prefix: Optional[str]  # None = auto (BGE models get their recommended prefix)
    # --- chunking (sizes in *approximate* tokens, see text_chunker.estimate_tokens) ---
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    csv_overlap_rows: int
    # --- retrieval ---
    top_k: int
    max_distance: Optional[float]  # None = no distance cut-off
    max_context_tokens: int
    # --- LLM ---
    llm_provider: str  # "groq" (default) or "anthropic"
    groq_model: str
    anthropic_model: str
    llm_max_tokens: int
    llm_timeout_s: float

    @classmethod
    def from_env(cls) -> "RAGSettings":
        _load_dotenv()
        settings = cls(
            documents_dir=_resolve_path(os.getenv("RAG_DOCUMENTS_DIR", "documents")),
            chroma_dir=_resolve_path(os.getenv("RAG_CHROMA_DIR", "chroma_db")),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "project_documents"),
            project_folder_level=_env_int("RAG_PROJECT_FOLDER_LEVEL", 2),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 32),
            embedding_device=os.getenv("RAG_EMBEDDING_DEVICE") or None,
            embedding_query_prefix=os.getenv("RAG_EMBEDDING_QUERY_PREFIX"),
            chunk_size_tokens=_env_int("RAG_CHUNK_SIZE_TOKENS", 350),
            chunk_overlap_tokens=_env_int("RAG_CHUNK_OVERLAP_TOKENS", 50),
            csv_overlap_rows=_env_int("RAG_CSV_OVERLAP_ROWS", 1),
            top_k=_env_int("RAG_TOP_K", 5),
            max_distance=_env_float_optional("RAG_MAX_DISTANCE"),
            max_context_tokens=_env_int("RAG_MAX_CONTEXT_TOKENS", 6000),
            llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", _env_int("ANTHROPIC_MAX_TOKENS", 1500)),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", _env_float("ANTHROPIC_TIMEOUT_S", 60.0)),
        )
        settings.validate()
        return settings

    def with_overrides(self, **changes: object) -> "RAGSettings":
        """Return a copy with some fields replaced (``None`` values are ignored) and re-validated."""
        changes = {k: v for k, v in changes.items() if v is not None}
        for key in ("documents_dir", "chroma_dir"):
            if key in changes:
                changes[key] = _resolve_path(str(changes[key]))
        new = replace(self, **changes)
        new.validate()
        return new

    def validate(self) -> None:
        if not _COLLECTION_NAME_RE.match(self.collection_name):
            raise ConfigError(
                "RAG_COLLECTION_NAME must be 3-512 characters from [a-zA-Z0-9._-], "
                f"starting and ending with an alphanumeric character. Got {self.collection_name!r}"
            )
        if self.chunk_size_tokens < 50:
            raise ConfigError("RAG_CHUNK_SIZE_TOKENS must be at least 50")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ConfigError("RAG_CHUNK_OVERLAP_TOKENS must be >= 0 and smaller than the chunk size")
        if self.csv_overlap_rows < 0:
            raise ConfigError("RAG_CSV_OVERLAP_ROWS must be >= 0")
        if self.top_k < 1:
            raise ConfigError("RAG_TOP_K must be >= 1")
        if self.embedding_batch_size < 1:
            raise ConfigError("RAG_EMBEDDING_BATCH_SIZE must be >= 1")
        if self.project_folder_level < 0:
            raise ConfigError("RAG_PROJECT_FOLDER_LEVEL must be >= 0")
        if self.llm_provider not in ("groq", "anthropic"):
            raise ConfigError(f"LLM_PROVIDER must be 'groq' or 'anthropic', got {self.llm_provider!r}")
        if self.llm_max_tokens < 100:
            raise ConfigError("LLM_MAX_TOKENS must be at least 100")
        if self.max_context_tokens < 200:
            raise ConfigError("RAG_MAX_CONTEXT_TOKENS must be at least 200")