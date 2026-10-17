"""ChromaDB persistent vector store.

* Uses the official ``chromadb`` package with ``PersistentClient``; the directory
  and collection name are configurable. The directory is **never** deleted by this
  module: :meth:`ChromaVectorStore.reset` only drops the named *collection*.
* Embeddings are generated outside Chroma (``embedding_function=None``). The
  embedding model name and dimension are written into the collection metadata at
  creation and verified on every later open, so vectors from different models
  can never be mixed in one collection.
* Distance metric is **cosine** (``distance = 1 - cosine_similarity``; 0 = identical
  direction, 1 = unrelated/orthogonal, 2 = opposite). ``similarity`` in results is
  ``1 - distance``. Neither is a probability or a confidence score: use them only to
  rank passages and, after calibration on your own documents, as a cut-off.
* Chunk ids are deterministic, so re-ingesting unchanged content is an upsert, not
  a duplicate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import chromadb

from .text_chunker import Chunk

logger = logging.getLogger(__name__)

_MAX_BATCH = 500
_SCAN_PAGE = 5000  # rows per page when scanning metadata


class VectorStoreError(RuntimeError):
    """Database error or configuration mismatch (wrong model/dimension)."""


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float  # cosine distance, lower = closer
    similarity: float  # 1 - distance (cosine similarity); NOT a probability


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce metadata to Chroma-compatible scalars (str/int/float/bool); drop ``None``."""
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            clean[str(key)] = value
        elif isinstance(value, (int, float, str)):
            clean[str(key)] = value
        elif isinstance(value, (list, tuple, set)):
            clean[str(key)] = ", ".join(str(v) for v in value)
        else:
            clean[str(key)] = str(value)
    return clean


def build_where(
    project_id: Optional[str] = None,
    project_ids: Optional[Sequence[str]] = None,
    doc_types: Optional[Sequence[str]] = None,
    source_paths: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a Chroma ``where`` filter. Returns ``None`` when no filter is requested.

    Callers must handle ``project_ids == []`` (no accessible projects) *before*
    calling this; an empty list means "nothing allowed", not "no restriction".
    """
    clauses: List[Dict[str, Any]] = []
    if project_id:
        clauses.append({"project_id": project_id})
    if project_ids is not None and len(project_ids) > 0:
        clauses.append({"project_id": {"$in": list(project_ids)}})
    if doc_types:
        clauses.append({"doc_type": {"$in": list(doc_types)}})
    if source_paths:
        clauses.append({"source_path": {"$in": list(source_paths)}})
    if extra:
        clauses.append(extra)
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaVectorStore:
    """Thin wrapper around one persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str = "project_documents",
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._open_collection()
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not open ChromaDB at {self.persist_dir}: {exc}") from exc

    # ------------------------------------------------------------------ setup
    @staticmethod
    def peek_metadata(persist_dir: Path | str, collection_name: str) -> Optional[Dict[str, Any]]:
        """Return an existing collection's metadata, or ``None`` if the directory/collection does not exist.

        Read-only: never creates a directory or collection. Used so ingestion can learn the stored
        embedding model/dimension without loading the (slow) embedding model first.
        """
        path = Path(persist_dir)
        if not path.is_dir():
            return None
        try:
            client = chromadb.PersistentClient(path=str(path))
            return dict(client.get_collection(name=collection_name, embedding_function=None).metadata or {})
        except Exception:
            return None

    def _open_collection(self):
        metadata: Dict[str, Any] = {}
        if self.embedding_model:
            metadata["embedding_model"] = self.embedding_model
        if self.embedding_dim:
            metadata["embedding_dim"] = int(self.embedding_dim)
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=metadata or None,
            embedding_function=None,
        )
        self._verify_compatibility(collection)
        return collection

    def _verify_compatibility(self, collection) -> None:
        stored = collection.metadata or {}
        stored_model = stored.get("embedding_model")
        stored_dim = stored.get("embedding_dim")
        if self.embedding_model and stored_model and stored_model != self.embedding_model:
            raise VectorStoreError(
                f"Collection {self.collection_name!r} was built with embedding model {stored_model!r}, "
                f"but {self.embedding_model!r} is configured. Use a new RAG_COLLECTION_NAME or reset the "
                "collection and re-ingest."
            )
        if self.embedding_dim and stored_dim and int(stored_dim) != int(self.embedding_dim):
            raise VectorStoreError(
                f"Collection {self.collection_name!r} stores {stored_dim}-dimensional vectors, "
                f"but the configured embedder produces {self.embedding_dim}. Use a new collection or reset it."
            )
        if self.embedding_dim is None and stored_dim:
            self.embedding_dim = int(stored_dim)
        if self.embedding_model is None and stored_model:
            self.embedding_model = str(stored_model)

    # ------------------------------------------------------------------ writes
    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], batch_size: int = _MAX_BATCH) -> int:
        """Insert or update chunks. Idempotent: the same ids overwrite instead of duplicating."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return 0
        dimension = self.embedding_dim or len(embeddings[0])
        for index, vector in enumerate(embeddings):
            if len(vector) != dimension:
                raise VectorStoreError(
                    f"Embedding {index} has dimension {len(vector)}; collection expects {dimension}"
                )
        ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        batch_size = max(1, min(batch_size, _MAX_BATCH))
        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                metadatas = []
                for chunk in batch:
                    meta = dict(chunk.metadata)
                    meta["ingested_at"] = ingested_at
                    if self.embedding_model:
                        meta["embedding_model"] = self.embedding_model
                    metadatas.append(normalize_metadata(meta))
                self._collection.upsert(
                    ids=[c.id for c in batch],
                    documents=[c.text for c in batch],
                    embeddings=[list(map(float, v)) for v in embeddings[start : start + batch_size]],
                    metadatas=metadatas,
                )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc
        if self.embedding_dim is None:
            self.embedding_dim = dimension
        return len(chunks)

    def delete_source(self, source_path: str, keep_ids: Optional[Iterable[str]] = None) -> int:
        """Delete chunks of one source file, optionally keeping ``keep_ids``. Returns number deleted."""
        keep: Set[str] = set(keep_ids or [])
        try:
            existing = self._collection.get(where={"source_path": source_path}, include=[])["ids"]
            stale = [i for i in existing if i not in keep]
            for start in range(0, len(stale), _MAX_BATCH):
                self._collection.delete(ids=stale[start : start + _MAX_BATCH])
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB delete failed for {source_path}: {exc}") from exc
        return len(stale)

    def delete_ids(self, ids: Iterable[str]) -> int:
        """Delete chunks by id (no lookup query needed when the caller already knows the ids)."""
        id_list = list(dict.fromkeys(ids))
        try:
            for start in range(0, len(id_list), _MAX_BATCH):
                self._collection.delete(ids=id_list[start : start + _MAX_BATCH])
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB delete failed: {exc}") from exc
        return len(id_list)

    def reset(self) -> None:
        """Drop and recreate the *collection* only. The persistence directory is left in place."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._open_collection()
        except Exception as exc:
            raise VectorStoreError(f"Could not reset collection {self.collection_name!r}: {exc}") from exc
        logger.warning("Collection %s was reset (all vectors removed)", self.collection_name)

    # ------------------------------------------------------------------ reads
    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB count failed: {exc}") from exc

    def source_chunk_ids(self, source_path: str) -> Set[str]:
        """Ids of all chunks currently stored for ``source_path``."""
        try:
            return set(self._collection.get(where={"source_path": source_path}, include=[])["ids"])
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB lookup failed for {source_path}: {exc}") from exc

    def _scan_metadatas(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield ``(chunk_id, metadata)`` for every chunk, one page at a time (bounded memory)."""
        offset = 0
        while True:
            page = self._collection.get(include=["metadatas"], limit=_SCAN_PAGE, offset=offset)
            ids = page.get("ids") or []
            if not ids:
                return
            metadatas = page.get("metadatas") or [None] * len(ids)
            for chunk_id, meta in zip(ids, metadatas):
                yield chunk_id, dict(meta or {})
            if len(ids) < _SCAN_PAGE:
                return
            offset += _SCAN_PAGE

    def source_index(self) -> Dict[str, Set[str]]:
        """Map ``source_path -> set of chunk ids`` for the whole collection in ONE pass.

        Replaces one ``get(where=...)`` query per file during ingestion.
        """
        index: Dict[str, Set[str]] = {}
        try:
            for chunk_id, meta in self._scan_metadatas():
                index.setdefault(str(meta.get("source_path", "(unknown)")), set()).add(chunk_id)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB read failed: {exc}") from exc
        return index

    def list_sources(self) -> Dict[str, int]:
        """Map ``source_path -> chunk count`` for everything indexed."""
        return dict(sorted((path, len(ids)) for path, ids in self.source_index().items()))

    def stats(self) -> Dict[str, Any]:
        sources: Dict[str, int] = {}
        projects: Dict[str, int] = {}
        try:
            for _, meta in self._scan_metadatas():  # single scan (was two full reads)
                path = meta.get("source_path", "(unknown)")
                project = meta.get("project_id", "(none)")
                sources[path] = sources.get(path, 0) + 1
                projects[project] = projects.get(project, 0) + 1
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB read failed: {exc}") from exc
        return {
            "persist_dir": str(self.persist_dir),
            "collection": self.collection_name,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "total_chunks": sum(sources.values()),
            "total_sources": len(sources),
            "chunks_per_project": dict(sorted(projects.items())),
            "chunks_per_source": dict(sorted(sources.items())),
        }

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Nearest-neighbour search. Returns ``[]`` for an empty collection or no matches."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.embedding_dim and len(query_embedding) != self.embedding_dim:
            raise VectorStoreError(
                f"Query embedding has dimension {len(query_embedding)}; collection expects {self.embedding_dim}"
            )
        try:
            if self._collection.count() == 0:
                return []
            raw = self._collection.query(
                query_embeddings=[list(map(float, query_embedding))],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        results: List[SearchResult] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            results.append(
                SearchResult(
                    id=chunk_id,
                    text=text or "",
                    metadata=dict(meta or {}),
                    distance=float(distance),
                    similarity=1.0 - float(distance),
                )
            )
        return results