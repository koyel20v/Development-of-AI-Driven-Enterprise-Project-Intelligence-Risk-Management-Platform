"""Shared embedding model holder + on-disk vector cache.

* :class:`ManagedEmbedder` loads the sentence-transformers model ONCE (lazily or in a background thread),
  serialises model calls with a lock and exposes the same interface as the plain embedder.
* :class:`EmbeddingCache` stores vectors in a small SQLite file. The cache key contains the **model name**,
  so vectors of one model can never be returned for another (e.g. after switching BGE -> MiniLM).
"""

from __future__ import annotations

import array
import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .embeddings import EmbeddingError, create_embedder

logger = logging.getLogger(__name__)


def _key(model_name: str, text: str) -> str:
    return hashlib.sha256(model_name.encode("utf-8") + b"\0" + text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, data BLOB NOT NULL)")
        self._db.commit()

    def get_many(self, model_name: str, texts: Sequence[str]) -> Dict[int, List[float]]:
        """Return ``{index: vector}`` for every text found in the cache."""
        found: Dict[int, List[float]] = {}
        keys = [_key(model_name, t) for t in texts]
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                marks = ",".join("?" * len(chunk))
                rows = {k: (d, b) for k, d, b in self._db.execute(f"SELECT key, dim, data FROM vec WHERE key IN ({marks})", chunk)}
                for offset, key in enumerate(chunk):
                    if key in rows:
                        dim, blob = rows[key]
                        values = array.array("f")
                        values.frombytes(blob)
                        if len(values) == dim:
                            found[start + offset] = values.tolist()
        return found

    def put_many(self, model_name: str, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        rows = [(_key(model_name, t), len(v), array.array("f", v).tobytes()) for t, v in zip(texts, vectors)]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO vec (key, dim, data) VALUES (?, ?, ?)", rows)
            self._db.commit()


@dataclass
class _Stats:
    batches: int = 0
    texts: int = 0
    seconds: float = 0.0


class ManagedEmbedder:
    """Lazy, thread-safe, cached wrapper around the configured embedder."""

    def __init__(self, settings, cache: Optional[EmbeddingCache] = None, known_dimension: Optional[int] = None) -> None:
        self._settings = settings
        self._cache = cache
        self._known_dimension = known_dimension
        self._inner = None
        self._load_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self.model_name = settings.embedding_model
        self.loading = False
        self.load_error: Optional[str] = None
        self.load_seconds = 0.0
        self.cache_hits = 0
        self.deduplicated = 0
        self.model_texts = 0
        self.model_seconds = 0.0

    # ------------------------------------------------------------------ loading
    @property
    def ready(self) -> bool:
        return self._inner is not None

    @property
    def dimension_known(self) -> bool:
        return self._inner is not None or self._known_dimension is not None

    @property
    def dimension(self) -> int:
        if self._inner is not None:
            return int(self._inner.dimension)
        if self._known_dimension is not None:
            return int(self._known_dimension)
        return int(self._ensure().dimension)

    @property
    def device(self) -> str:
        model = getattr(self._inner, "_model", None)
        return str(getattr(model, "device", "cpu"))

    @property
    def max_seq_length(self) -> Optional[int]:
        return getattr(self._inner, "max_seq_length", None)

    @property
    def stats(self):
        inner = getattr(self._inner, "stats", None)
        return inner if inner is not None else _Stats()

    def _ensure(self):
        if self._inner is not None:
            return self._inner
        with self._load_lock:
            if self._inner is None:
                self.loading, self.load_error = True, None
                started = time.perf_counter()
                try:
                    self._inner = create_embedder(self._settings)
                    self.load_seconds = time.perf_counter() - started
                    self._known_dimension = int(self._inner.dimension)
                except Exception as exc:
                    self.load_error = str(exc)
                    raise
                finally:
                    self.loading = False
        return self._inner

    def warm_up_async(self) -> None:
        """Start loading the model in the background (no-op if loaded or already loading)."""
        if self._inner is not None or self.loading:
            return

        def run() -> None:
            try:
                self._ensure()
            except Exception as exc:
                logger.error("Embedding model could not be loaded: %s", exc)

        self.loading = True
        threading.Thread(target=run, name="embedder-warmup", daemon=True).start()

    # ------------------------------------------------------------------ embedding
    def counters(self) -> Dict[str, float]:
        return {
            "model_texts": self.model_texts, "model_seconds": self.model_seconds,
            "cache_hits": self.cache_hits, "deduplicated": self.deduplicated,
        }

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        inner = self._ensure()
        vectors: List[Optional[List[float]]] = [None] * len(texts)
        cached = self._cache.get_many(self.model_name, texts) if self._cache else {}
        for index, vector in cached.items():
            vectors[index] = vector
        self.cache_hits += len(cached)
        first_seen: Dict[str, int] = {}
        todo: List[int] = []
        for index, text in enumerate(texts):
            if vectors[index] is not None:
                continue
            if text in first_seen:  # identical text inside this call: embed once
                self.deduplicated += 1
                continue
            first_seen[text] = index
            todo.append(index)
        if todo:
            started = time.perf_counter()
            with self._call_lock:
                fresh = inner.embed_documents([texts[i] for i in todo])
            self.model_seconds += time.perf_counter() - started
            self.model_texts += len(todo)
            for index, vector in zip(todo, fresh):
                vectors[index] = vector
            if self._cache:
                try:
                    self._cache.put_many(self.model_name, [texts[i] for i in todo], fresh)
                except Exception as exc:  # the cache is an optimisation only
                    logger.warning("Embedding cache write failed: %s", exc)
        for index, text in enumerate(texts):
            if vectors[index] is None:
                vectors[index] = vectors[first_seen[text]]
        if any(v is None for v in vectors):
            raise EmbeddingError("Internal error: some vectors were not produced")
        return vectors  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        inner = self._ensure()
        with self._call_lock:
            return inner.embed_query(text)
