"""RAG pipeline: ingestion, retrieval and grounded question answering.

Ingestion (``RAGPipeline.ingest``) and answering (``RAGPipeline.ask``) are separate
code paths; ingestion never needs an LLM key and answering never touches the
documents directory.

Access control
--------------
``retrieve``/``ask`` accept ``project_ids``: the list of projects the *calling user
may see*. The backend must derive it from its own authentication/authorisation
layer, never from untrusted request input. ``project_ids=None`` means "no project
restriction" (use only for trusted/admin callers); ``project_ids=[]`` means "no
access" and always returns no evidence.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .config import RAGSettings
from .document_loader import DEFAULT_MAX_FILE_MB, SUPPORTED_EXTENSIONS, LoadStats, _file_sha256, _iter_files
from .embeddings import Embedder, EmbeddingError, create_embedder
from .ingest_manifest import (
    STATUS_EMPTY,
    STATUS_OK,
    IngestManifest,
    ManifestEntry,
    compute_signature,
    manifest_path,
)
from .ingest_worker import PreparedFile, prepare_file
from .llm import INSUFFICIENT_SENTINEL, AnswerGenerator, LLMError, create_generator, describe_location
from .text_chunker import Chunk, ChunkingConfig, estimate_tokens
from .vector_store import ChromaVectorStore, SearchResult, VectorStoreError, build_where

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
# Default number of chunks (from several documents) embedded together; override with RAG_EMBED_FLUSH_CHUNKS.
EMBED_FLUSH_CHUNKS = 256
# progress(stage, done, total, detail): stages are "plan", "file" and "embed" (see RAGPipeline.ingest)
ProgressCallback = Callable[[str, int, int, str], None]
NO_EVIDENCE_ANSWER = (
    f"{INSUFFICIENT_SENTINEL}: no relevant passages were found in the indexed project documents "
    "for this question (and filters)."
)
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# --------------------------------------------------------------------------- result types
@dataclass
class Source:
    """One retrieved passage. ``citation_id`` matches ``[n]`` markers in the answer."""

    citation_id: int
    chunk_id: str
    source_file: str
    source_path: str
    doc_type: str
    project_id: str
    page_number: Optional[int]
    section: Optional[str]
    row_range: Optional[str]
    location: str
    distance: float  # cosine distance (lower = closer). Not a probability.
    similarity: float  # 1 - distance. Not a confidence score.
    text: str
    cited: bool  # True if the answer cites this source
    metadata: Dict[str, Any]


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sufficient_evidence: bool
    sources: List[Source]
    filters_applied: Dict[str, Any]
    model: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timings_ms: Dict[str, int] = field(default_factory=dict)

    @property
    def cited_sources(self) -> List[Source]:
        return [s for s in self.sources if s.cited]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionReport:
    documents_loaded: int = 0
    documents_unchanged: int = 0
    documents_indexed: int = 0
    chunks_generated: int = 0
    chunks_embedded: int = 0
    records_upserted: int = 0
    stale_chunks_deleted: int = 0
    pruned_chunks: int = 0
    failures: List[Tuple[str, str]] = field(default_factory=list)
    load_stats: Dict[str, int] = field(default_factory=dict)
    unsupported_files: List[str] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)
    total_chunks_in_store: int = 0
    duration_s: float = 0.0
    # --- performance reporting (added; the fields above are unchanged) ---
    mode: str = ""  # "initial indexing" | "incremental" | "full re-index (--force)"
    files_discovered: int = 0  # supported files found on disk
    documents_skipped_without_parsing: int = 0  # unchanged per manifest: not even opened for parsing
    documents_empty_cached: int = 0  # known-empty files (e.g. scanned PDFs) not re-parsed
    embed_calls: int = 0  # calls to Embedder.embed_documents
    model_batches: int = 0  # model.encode batches actually executed (local model only)
    manifest_note: str = ""
    timings_s: Dict[str, float] = field(default_factory=dict)  # hash/load/chunk/embed/db; load+chunk are summed across workers

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Pending:
    """A document waiting for its chunks to be embedded and written."""

    path: str
    sha: str
    to_embed: List[Chunk]
    new_ids: Set[str]
    stale_ids: List[str]
    total_chunks: int


def _source_from_result(number: int, result: SearchResult, cited: bool = False) -> Source:
    meta = result.metadata
    page = int(meta.get("page_number") or 0)
    start, end = int(meta.get("row_start") or 0), int(meta.get("row_end") or 0)
    return Source(
        citation_id=number,
        chunk_id=result.id,
        source_file=str(meta.get("source_file", "")),
        source_path=str(meta.get("source_path", "")),
        doc_type=str(meta.get("doc_type", "")),
        project_id=str(meta.get("project_id", "")),
        page_number=page or None,
        section=str(meta.get("section") or "") or None,
        row_range=(None if start <= 0 else (f"{start}" if start == end else f"{start}-{end}")),
        location=describe_location(meta),
        distance=result.distance,
        similarity=result.similarity,
        text=result.text,
        cited=cited,
        metadata=dict(meta),
    )


def parse_citations(answer: str, max_id: int) -> Set[int]:
    """Return the valid ``[n]`` citation numbers used in ``answer`` (1..max_id)."""
    used: Set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if 1 <= number <= max_id:
                used.add(number)
    return used


# --------------------------------------------------------------------------- pipeline
class RAGPipeline:
    """Orchestrates loading, chunking, embedding, ChromaDB storage and answer generation.

    Providers are created lazily and can be injected (used by the tests):

    >>> pipeline = RAGPipeline()                 # configuration from environment / rag/.env
    >>> pipeline.ingest()                        # index documents/
    >>> pipeline.ask("What are the major risks of Project Alpha?", project_ids=["project_alpha"])
    """

    def __init__(
        self,
        settings: Optional[RAGSettings] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[ChromaVectorStore] = None,
        generator: Optional[AnswerGenerator] = None,
    ) -> None:
        self.settings = settings or RAGSettings.from_env()
        self._embedder = embedder
        self._store = store
        self._generator = generator

    # ------------------------------------------------------------------ lazy providers
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(self.settings)
        return self._embedder

    @property
    def store(self) -> ChromaVectorStore:
        if self._store is None:
            embedder = self._embedder
            self._store = ChromaVectorStore(
                persist_dir=self.settings.chroma_dir,
                collection_name=self.settings.collection_name,
                embedding_model=embedder.model_name if embedder else self.settings.embedding_model,
                embedding_dim=embedder.dimension if embedder else None,
            )
        return self._store

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = create_generator(self.settings)
        return self._generator

    def _ready_store(self) -> ChromaVectorStore:
        """Return the store after confirming it matches the active embedder."""
        embedder = self.embedder  # load the embedder first so the store knows model + dimension
        store = self.store
        if store.embedding_dim and store.embedding_dim != embedder.dimension:
            raise VectorStoreError(
                f"Collection {store.collection_name!r} holds {store.embedding_dim}-dimensional vectors but the "
                f"embedder {embedder.model_name!r} produces {embedder.dimension}. Use a new collection or reset it."
            )
        if store.embedding_model and store.embedding_model != embedder.model_name:
            raise VectorStoreError(
                f"Collection {store.collection_name!r} was built with {store.embedding_model!r}, "
                f"not {embedder.model_name!r}. Use a new collection or reset it."
            )
        if store.embedding_dim is None:
            store.embedding_dim = embedder.dimension
        if store.embedding_model is None:
            store.embedding_model = embedder.model_name
        return store

    # ------------------------------------------------------------------ ingestion
    def _store_for_ingest(self) -> ChromaVectorStore:
        """Open the store, loading the embedding model only if it is really needed.

        Importing torch and loading the model takes seconds and used to happen on EVERY run, even when
        nothing needed embedding. If the collection already records its embedding model and dimension,
        the store can be opened from that metadata and the model is loaded lazily on first use.
        """
        if self._embedder is not None or self._store is not None:
            return self._ready_store()
        meta = ChromaVectorStore.peek_metadata(self.settings.chroma_dir, self.settings.collection_name)
        if meta and meta.get("embedding_dim"):
            # The constructor verifies the stored model name against the configured one.
            self._store = ChromaVectorStore(
                persist_dir=self.settings.chroma_dir,
                collection_name=self.settings.collection_name,
                embedding_model=self.settings.embedding_model,
                embedding_dim=int(meta["embedding_dim"]),
            )
            return self._store
        return self._ready_store()  # new collection: the embedder's model/dimension are needed to create it

    def _open_manifest(self, force: bool) -> IngestManifest:
        model_name = self._embedder.model_name if self._embedder is not None else self.settings.embedding_model
        path = manifest_path(self.settings.chroma_dir, self.settings.collection_name)
        signature = compute_signature(self.settings, model_name)
        if self.settings.use_manifest and not force:
            return IngestManifest.load(path, signature)
        manifest = IngestManifest(path, signature)
        manifest.note = "not used (--force)" if force else "disabled (RAG_USE_MANIFEST=false)"
        return manifest

    def reset_index(self) -> None:
        """Drop the collection (never the directory) and forget the ingestion manifest."""
        self._ready_store().reset()
        IngestManifest(
            manifest_path(self.settings.chroma_dir, self.settings.collection_name), signature=""
        ).delete_file()

    def _iter_prepared(
        self, work: List[Tuple[Path, str, str]], root: Path, config: ChunkingConfig
    ) -> Iterator[Tuple[str, str, PreparedFile]]:
        """Parse + chunk files, in order. Uses a bounded process pool when ``load_workers > 1``."""
        level, max_mb = self.settings.project_folder_level, DEFAULT_MAX_FILE_MB

        def run(path: Path, relative: str) -> PreparedFile:
            return prepare_file(str(path), str(root), relative, level, max_mb, config)

        workers = min(self.settings.load_workers, len(work))
        done: Set[str] = set()
        if workers > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=workers) as pool:
                    iterator = iter(work)
                    window: deque = deque()  # bounded look-ahead keeps memory use flat

                    def submit(item: Tuple[Path, str, str]) -> None:
                        path, relative, sha = item
                        window.append((relative, sha, pool.submit(prepare_file, str(path), str(root), relative, level, max_mb, config)))

                    for item in islice(iterator, workers * 2):
                        submit(item)
                    while window:
                        relative, sha, future = window.popleft()
                        prepared = future.result()
                        done.add(relative)
                        yield relative, sha, prepared
                        following = next(iterator, None)
                        if following is not None:
                            submit(following)
            except Exception as exc:  # pool could not start / worker died: finish the rest in-process
                logger.warning("Parallel loading failed (%s: %s); continuing in a single process", type(exc).__name__, exc)
        for path, relative, sha in work:
            if relative not in done:
                yield relative, sha, run(path, relative)

    def ingest(
        self,
        documents_dir: Optional[Path | str] = None,
        force: bool = False,
        prune: bool = False,
        progress: Optional[ProgressCallback] = None,
        only_paths: Optional[Sequence[str]] = None,
    ) -> IngestionReport:
        """Load, chunk, embed and index all documents. Safe to run repeatedly.

        Workflow
        --------
        1. One bulk read of the ChromaDB metadata (``source_path -> chunk ids``).
        2. Every file is hashed (SHA-256). A file whose hash matches the manifest AND whose chunk count
           in ChromaDB matches is skipped without being parsed.
        3. Only new/changed files are parsed and chunked (optionally in ``RAG_LOAD_WORKERS`` processes).
           Chunk ids are deterministic, so chunks already stored are never re-embedded.
        4. New chunks of many documents are embedded in one batch call, written to ChromaDB in one upsert,
           and stale chunks of changed files are deleted by id. The manifest is saved after each batch,
           so an interrupted run resumes where it stopped.

        ``only_paths`` (relative POSIX paths below the documents directory) restricts the run to exactly those
        files: nothing else in the folder is hashed, parsed or embedded, and ``prune`` is ignored. The upload UI
        must always pass it so historical files are never indexed as a side effect of an upload.

        ``force=True`` re-embeds everything and ignores the manifest. ``prune=True`` also removes indexed
        sources whose files no longer exist. The persistence directory itself is never deleted.
        ``progress(stage, done, total, detail)`` is optional: stage ``"plan"`` (total = files to process),
        ``"file"`` (done of total files processed) and ``"embed"`` (done = chunks embedded so far).
        """
        started = time.perf_counter()
        report = IngestionReport()
        root = Path(documents_dir) if documents_dir else self.settings.documents_dir
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Documents directory not found: {root}")
        notify: ProgressCallback = progress or (lambda *_: None)
        config = ChunkingConfig(
            chunk_size_tokens=self.settings.chunk_size_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
            csv_overlap_rows=self.settings.csv_overlap_rows,
        )
        timers = {"hash": 0.0, "load": 0.0, "chunk": 0.0, "embed": 0.0, "db": 0.0}
        store = self._store_for_ingest()
        manifest = self._open_manifest(force)
        report.manifest_note = manifest.note

        t0 = time.perf_counter()
        index = store.source_index()  # ONE query instead of one per file
        timers["db"] += time.perf_counter() - t0
        stored_chunks = sum(len(ids) for ids in index.values())
        report.mode = "full re-index (--force)" if force else ("initial indexing" if stored_chunks == 0 else "incremental")

        # ---- discovery + change detection (no parsing yet)
        load_stats = LoadStats()
        jobs: List[Tuple[Path, str]] = []
        allowed = {Path(p).as_posix().lstrip("/") for p in only_paths} if only_paths is not None else None
        if allowed is not None:
            prune = False  # a restricted run must never delete anything it was not asked about
            report.mode = f"{report.mode} (selected files only)"
        for path in _iter_files(root, True):
            relative = path.relative_to(root).as_posix()
            if allowed is not None and relative not in allowed:
                continue
            load_stats.files_found += 1
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                load_stats.unsupported_files.append(relative)
                logger.warning("Skipping unsupported file type: %s", relative)
                continue
            load_stats.discovered_paths.append(relative)
            jobs.append((path, relative))
        report.files_discovered = len(jobs)

        to_parse: List[Tuple[Path, str, str]] = []  # (path, relative path, sha256)
        for path, relative in jobs:
            t0 = time.perf_counter()
            try:
                oversized = path.stat().st_size / (1024 * 1024) > DEFAULT_MAX_FILE_MB
                sha = "" if oversized else _file_sha256(path)  # oversized files are rejected by the loader
            except OSError as exc:
                load_stats.failures.append((relative, f"cannot read file: {exc}"))
                continue
            finally:
                timers["hash"] += time.perf_counter() - t0
            entry = manifest.get(relative)
            if entry is not None and sha and entry.sha256 == sha:
                if entry.status == STATUS_EMPTY:
                    load_stats.empty_files.append(relative)
                    report.documents_empty_cached += 1
                    continue
                if entry.status == STATUS_OK and len(index.get(relative, ())) == entry.chunks:
                    report.documents_unchanged += 1
                    report.documents_skipped_without_parsing += 1
                    load_stats.documents_loaded += 1
                    continue
            to_parse.append((path, relative, sha))
        logger.info(
            "Ingestion plan (%s): %d supported files, %d unchanged (not parsed), %d to process. Manifest: %s",
            report.mode, len(jobs), report.documents_unchanged + report.documents_empty_cached, len(to_parse),
            manifest.note or "in use",
        )
        notify("plan", len(to_parse), len(jobs), report.mode)

        # ---- parse -> embed -> store
        pending: List[_Pending] = []
        pending_chunks = 0
        embedder_checked = False

        def use_embedder() -> Embedder:
            """Load the model on first need, then verify it matches the collection."""
            nonlocal embedder_checked
            embedder = self.embedder
            self._ready_store()
            if not embedder_checked:
                embedder_checked = True
                max_seq = getattr(embedder, "max_seq_length", None)
                if max_seq and config.chunk_size_tokens > max_seq:
                    logger.warning(
                        "chunk size (%d est. tokens) exceeds the embedding model's limit (%d); chunk tails will be "
                        "truncated during embedding. Lower RAG_CHUNK_SIZE_TOKENS.",
                        config.chunk_size_tokens,
                        max_seq,
                    )
            return embedder

        def commit(ready: List[Tuple[_Pending, List[List[float]]]]) -> None:
            if not ready:
                return
            t_db = time.perf_counter()
            chunks = [c for item, _ in ready for c in item.to_embed]
            vectors = [v for _, vecs in ready for v in vecs]
            written: List[Tuple[_Pending, List[List[float]]]] = []
            try:  # one upsert for many documents = far fewer SQLite transactions / HNSW writes
                if chunks:
                    report.records_upserted += store.upsert_chunks(chunks, vectors)
                written = list(ready)
            except VectorStoreError as exc:
                logger.warning("Batched write failed (%s); retrying document by document", exc)
                for item, vecs in ready:
                    try:
                        if item.to_embed:
                            report.records_upserted += store.upsert_chunks(item.to_embed, vecs)
                        written.append((item, vecs))
                    except VectorStoreError as exc2:
                        report.failures.append((item.path, str(exc2)))
                        logger.error("Indexing failed for %s: %s", item.path, exc2)
            for item, _ in written:
                try:
                    if item.stale_ids:
                        store.delete_ids(item.stale_ids)  # ids already known: no lookup query
                except VectorStoreError as exc:
                    report.failures.append((item.path, str(exc)))
                    logger.error("Could not delete stale chunks of %s: %s", item.path, exc)
                    continue
                report.chunks_embedded += len(item.to_embed)
                report.stale_chunks_deleted += len(item.stale_ids)
                report.documents_indexed += 1
                index[item.path] = set(item.new_ids)
                manifest.record(item.path, ManifestEntry(item.sha, STATUS_OK, item.total_chunks))
            timers["db"] += time.perf_counter() - t_db
            if self.settings.use_manifest:
                manifest.save()  # checkpoint: an interrupted run resumes from here

        def flush() -> None:
            nonlocal pending, pending_chunks
            if not pending:
                return
            batch, pending, pending_chunks = pending, [], 0
            texts = [c.text for item in batch for c in item.to_embed]
            ready: List[Tuple[_Pending, List[List[float]]]] = []
            t_embed = time.perf_counter()
            if not texts:  # only stale chunks to delete: no model needed
                ready = [(item, []) for item in batch]
            else:
                embedder = use_embedder()
                try:
                    report.embed_calls += 1
                    all_vectors = embedder.embed_documents(texts)  # one call for many documents
                    offset = 0
                    for item in batch:
                        ready.append((item, all_vectors[offset : offset + len(item.to_embed)]))
                        offset += len(item.to_embed)
                except EmbeddingError:
                    # Isolate the bad document: retry one by one so healthy files are still indexed.
                    logger.warning("Batch embedding failed; retrying document by document")
                    ready = []
                    for item in batch:
                        if not item.to_embed:
                            ready.append((item, []))
                            continue
                        try:
                            report.embed_calls += 1
                            ready.append((item, embedder.embed_documents([c.text for c in item.to_embed])))
                        except EmbeddingError as exc:
                            report.failures.append((item.path, str(exc)))
                            logger.error("Indexing failed for %s: %s", item.path, exc)
            timers["embed"] += time.perf_counter() - t_embed
            commit(ready)
            logger.info("Progress: %d chunks embedded so far", report.chunks_embedded)
            notify("embed", report.chunks_embedded, 0, "")

        flush_at = self.settings.embed_flush_chunks
        for number, (relative, sha, prepared) in enumerate(self._iter_prepared(to_parse, root, config), start=1):
            notify("file", number, len(to_parse), relative)
            timers["load"] += prepared.load_s
            timers["chunk"] += prepared.chunk_s
            load_stats.pages_failed += prepared.pages_failed
            if prepared.status == "failed":
                load_stats.failures.append((relative, prepared.error))
                logger.warning("Failed to load %s: %s", relative, prepared.error)
                continue
            if prepared.status == "empty":
                load_stats.empty_files.append(relative)
                logger.warning("Empty document %s: %s", relative, prepared.error)
                if sha:
                    manifest.record(relative, ManifestEntry(sha, STATUS_EMPTY, 0))
                continue
            load_stats.documents_loaded += 1
            load_stats.sections_loaded += prepared.sections
            chunks = prepared.chunks
            report.chunks_generated += len(chunks)
            new_ids = {c.id for c in chunks}
            existing = index.get(relative, set())
            if not force and existing == new_ids:  # parsed, but identical to what is stored
                report.documents_unchanged += 1
                manifest.record(relative, ManifestEntry(sha, STATUS_OK, len(chunks)))
                continue
            to_embed = chunks if force else [c for c in chunks if c.id not in existing]
            pending.append(_Pending(relative, sha, to_embed, new_ids, sorted(existing - new_ids), len(chunks)))
            pending_chunks += len(to_embed)
            if pending_chunks >= flush_at:
                flush()
        flush()
        logger.info(
            "Timing: hashing %.1fs | loading %.1fs | chunking %.1fs | embedding %.1fs | database %.1fs",
            timers["hash"], timers["load"], timers["chunk"], timers["embed"], timers["db"],
        )

        if prune:
            on_disk = set(load_stats.discovered_paths)
            for indexed_path in list(index):
                if indexed_path not in on_disk:
                    report.pruned_chunks += store.delete_ids(index.pop(indexed_path))
                    logger.info("Pruned %s (file no longer on disk)", indexed_path)
            for known_path in list(manifest.entries):
                if known_path not in on_disk:
                    manifest.remove(known_path)
        if self.settings.use_manifest:
            manifest.save()

        report.documents_loaded = load_stats.documents_loaded
        report.failures = [(p, m) for p, m in load_stats.failures] + report.failures
        report.unsupported_files = list(load_stats.unsupported_files)
        report.empty_files = list(load_stats.empty_files)
        report.load_stats = load_stats.summary()
        t0 = time.perf_counter()
        report.total_chunks_in_store = store.count()
        timers["db"] += time.perf_counter() - t0
        stats = getattr(self._embedder, "stats", None)
        report.model_batches = int(getattr(stats, "batches", 0))
        report.timings_s = {name: round(value, 3) for name, value in timers.items()}
        report.duration_s = round(time.perf_counter() - started, 2)
        logger.info("Ingestion finished: %s", {k: v for k, v in report.to_dict().items() if isinstance(v, (int, float))})
        return report

    # ------------------------------------------------------------------ retrieval
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        project_id: Optional[str] = None,
        project_ids: Optional[Sequence[str]] = None,
        doc_types: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        max_distance: Optional[float] = None,
    ) -> List[SearchResult]:
        """Return the closest chunks (closest first). Raises on provider/database errors."""
        question = self._validate_question(question)
        if project_ids is not None and len(project_ids) == 0:
            return []  # caller has access to no projects
        if project_id and project_ids is not None and project_id not in project_ids:
            return []  # requested project is outside the caller's allowed set
        store = self._ready_store()
        where = build_where(
            project_id=project_id, project_ids=project_ids, doc_types=doc_types, extra=filters
        )
        query_vector = self.embedder.embed_query(question)
        results = store.query(query_vector, top_k or self.settings.top_k, where=where)
        limit = max_distance if max_distance is not None else self.settings.max_distance
        if limit is not None:
            results = [r for r in results if r.distance <= limit]
        return results

    # ------------------------------------------------------------------ question answering
    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        project_id: Optional[str] = None,
        project_ids: Optional[Sequence[str]] = None,
        doc_types: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        max_distance: Optional[float] = None,
    ) -> RAGAnswer:
        """Answer ``question`` from the indexed documents. Provider failures are returned, not raised.

        Raises:
            ValueError: for an empty or over-long question (a programming/input error).
        """
        question = self._validate_question(question)
        filters_applied = {
            "project_id": project_id,
            "project_ids": list(project_ids) if project_ids is not None else None,
            "doc_types": list(doc_types) if doc_types else None,
            "extra": filters,
            "top_k": top_k or self.settings.top_k,
            "max_distance": max_distance if max_distance is not None else self.settings.max_distance,
        }
        timings: Dict[str, int] = {}

        started = time.perf_counter()
        try:
            results = self.retrieve(question, top_k, project_id, project_ids, doc_types, filters, max_distance)
        except (EmbeddingError, VectorStoreError) as exc:
            logger.error("Retrieval failed: %s", exc)
            return RAGAnswer(
                question=question,
                answer="The documents could not be searched because of a retrieval error. Please try again later.",
                sufficient_evidence=False,
                sources=[],
                filters_applied=filters_applied,
                error=f"retrieval failed: {exc}",
                timings_ms={"retrieval": int((time.perf_counter() - started) * 1000)},
            )
        timings["retrieval"] = int((time.perf_counter() - started) * 1000)

        if not results:
            return RAGAnswer(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                sufficient_evidence=False,
                sources=[],
                filters_applied=filters_applied,
                timings_ms=timings,
            )

        results = self._fit_context(results)
        sources = [_source_from_result(i, r) for i, r in enumerate(results, start=1)]

        started = time.perf_counter()
        try:
            generation = self.generator.generate(question, results)
        except LLMError as exc:
            return RAGAnswer(
                question=question,
                answer="Relevant passages were found, but the answer could not be generated because the LLM "
                "provider failed. The retrieved evidence is listed in `sources`.",
                sufficient_evidence=False,
                sources=sources,
                filters_applied=filters_applied,
                error=f"generation failed: {exc}",
                timings_ms=timings,
            )
        timings["generation"] = int((time.perf_counter() - started) * 1000)

        answer = generation.text
        warnings: List[str] = []
        sufficient = not answer.lstrip().upper().startswith(INSUFFICIENT_SENTINEL)
        if sufficient:
            cited = parse_citations(answer, len(sources))
            for source in sources:
                source.cited = source.citation_id in cited
            if not cited:
                warnings.append("The answer contains no valid [n] citations; treat it as unverified.")
        if generation.stop_reason == "max_tokens":
            warnings.append("The answer was cut off at the token limit (raise LLM_MAX_TOKENS).")

        return RAGAnswer(
            question=question,
            answer=answer,
            sufficient_evidence=sufficient,
            sources=sources,
            filters_applied=filters_applied,
            model=generation.model,
            warnings=warnings,
            timings_ms=timings,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question is longer than {MAX_QUESTION_CHARS} characters")
        return question

    def _fit_context(self, results: List[SearchResult]) -> List[SearchResult]:
        """Keep the best-ranked passages that fit ``max_context_tokens`` (always at least one)."""
        kept: List[SearchResult] = []
        used = 0
        for result in results:
            tokens = estimate_tokens(result.text)
            if kept and used + tokens > self.settings.max_context_tokens:
                break
            kept.append(result)
            used += tokens
        return kept