"""Upload handling and background indexing jobs for the Streamlit app.

Indexing only ever starts from an explicit call to :meth:`IngestManager.start` and only ever touches the files
the user selected (``RAGPipeline.ingest(only_paths=...)``). Nothing here scans the documents folder.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .document_loader import DEFAULT_MAX_FILE_MB, SUPPORTED_EXTENSIONS
from .ingest_manifest import IngestManifest

_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass
class UploadItem:
    name: str
    relative: str  # POSIX path below the documents directory
    path: Path
    size: int
    sha256: str
    status: str = "ready"  # "ready" | "unsupported" | "empty" | "too large" | "duplicate (not indexed)"
    detail: str = ""


def _safe_name(name: str) -> str:
    base = _UNSAFE.sub("_", Path(name).name).strip(" .") or "upload"
    return base


def prepare_uploads(uploaded_files, dest_dir: Path, documents_root: Path, known_hashes: Optional[Dict[str, str]] = None) -> List[UploadItem]:
    """Validate and save the selected files below ``dest_dir``. Nothing is parsed or embedded here.

    ``known_hashes`` maps ``sha256 -> relative path`` of files already indexed; a file whose content is already
    indexed under a different path is reported as a duplicate and NOT written or indexed again.
    """
    dest_dir, documents_root = Path(dest_dir), Path(documents_root).resolve()
    known_hashes = known_hashes or {}
    items: List[UploadItem] = []
    seen: Dict[str, str] = {}
    for upload in uploaded_files:
        data = upload.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        name = _safe_name(upload.name)
        target = (dest_dir / name).resolve()
        try:
            relative = target.relative_to(documents_root).as_posix()
        except ValueError:
            relative = name
        item = UploadItem(upload.name, relative, target, len(data), sha)
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            item.status, item.detail = "unsupported", f"file type {extension or '(none)'} is not supported"
        elif not data:
            item.status, item.detail = "empty", "the file is empty"
        elif len(data) / 1_048_576 > DEFAULT_MAX_FILE_MB:
            item.status, item.detail = "too large", f"over the {DEFAULT_MAX_FILE_MB} MB limit"
        elif sha in seen:
            item.status, item.detail = "duplicate (not indexed)", f"same content as {seen[sha]} in this selection"
        elif sha in known_hashes and known_hashes[sha] != relative:
            item.status, item.detail = "duplicate (not indexed)", f"already indexed as {known_hashes[sha]}"
        if item.status == "ready":
            seen[sha] = upload.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                target.write_bytes(data)  # same name, new content => replaced; the pipeline swaps the stale chunks
        items.append(item)
    return items


class _Cancelled(Exception):
    pass


_ids = itertools.count(1)


class IngestJob:
    def __init__(self, label: str) -> None:
        self.id = next(_ids)
        self.label = label
        self._lock = threading.Lock()
        self._cancel = False
        self._started = time.perf_counter()
        self._ended: Optional[float] = None
        self._d: Dict[str, Any] = {
            "status": "pending", "stage": "opening the index", "files_total": 0, "files_done": 0, "current_file": "",
            "chunks_committed": 0, "error": "", "error_detail": "", "report": None, "file_results": [],
            "model_load_s": 0.0, "model_texts": 0, "model_seconds": 0.0, "cache_hits": 0, "deduplicated": 0,
        }

    @property
    def status(self) -> str:
        return self._d["status"]

    def cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def update(self, **values: Any) -> None:
        with self._lock:
            self._d.update(values)

    def finish(self) -> None:
        self._ended = time.perf_counter()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = dict(self._d)
        snap["label"] = self.label
        snap["elapsed_s"] = round((self._ended or time.perf_counter()) - self._started, 2)
        return snap


class IngestManager:
    """Runs one indexing job at a time in a background thread."""

    def __init__(self, pipeline, embedder) -> None:
        self.pipeline = pipeline
        self.embedder = embedder
        self.current: Optional[IngestJob] = None
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.current is not None and self.current.status in ("pending", "running")

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def indexed_hashes(self) -> Dict[str, str]:
        try:
            return self.pipeline._open_manifest(False).indexed_hashes()
        except Exception:
            return {}

    def start(self, uploads: List[UploadItem]) -> IngestJob:
        with self._lock:
            if self.busy:
                raise RuntimeError("An indexing job is already running")
            job = IngestJob(f"Run {len(self._history) + 1}")
            self.current = job
        threading.Thread(target=self._run, args=(job, list(uploads)), name=f"ingest-{job.id}", daemon=True).start()
        return job

    # ------------------------------------------------------------------ worker thread
    def _run(self, job: IngestJob, items: List[UploadItem]) -> None:
        job.update(status="running")
        ready = [i for i in items if i.status == "ready"]
        was_ready = self.embedder.ready
        before_counters = self.embedder.counters()
        try:
            store = self.pipeline._store_for_ingest()
            before = {i.relative: set(store.source_chunk_ids(i.relative)) for i in ready}
            job.update(stage="scanning & hashing files")

            def live() -> Dict[str, Any]:
                now = self.embedder.counters()
                return {k: now[k] - before_counters[k] for k in now}

            def progress(stage: str, done: int, total: int, detail: str) -> None:
                if job.cancelled:
                    raise _Cancelled()
                if stage == "plan":
                    job.update(files_total=done, stage="extracting text & chunking" if done else "nothing to parse (all files unchanged)")
                elif stage == "file":
                    job.update(files_done=done, files_total=total, current_file=detail, **live())
                elif stage == "embed":
                    job.update(stage="embedding", chunks_committed=done, **live())
                if self.embedder.loading and not self.embedder.ready:
                    job.update(stage="loading embedding model")

            report = self.pipeline.ingest(only_paths=[i.relative for i in ready], progress=progress)
            job.update(stage="finalising", **live())
            failures = dict(report.failures)
            empties = set(report.empty_files)
            rows = []
            for item in items:
                row = {"name": item.name, "relative": item.relative, "size": item.size, "sha256": item.sha256[:12],
                       "chunks": 0, "status": item.status, "detail": item.detail}
                if item.status == "ready":
                    after = set(store.source_chunk_ids(item.relative))
                    row["chunks"] = len(after)
                    if item.relative in failures:
                        row["status"], row["detail"] = "failed", failures[item.relative]
                    elif item.relative in empties:
                        row["status"], row["detail"] = "no text", "no extractable text (a scanned PDF needs OCR)"
                    elif not after:
                        row["status"], row["detail"] = "failed", "nothing was stored"
                    elif not before[item.relative]:
                        row["status"] = "indexed (new)"
                    elif before[item.relative] == after:
                        row["status"], row["detail"] = "unchanged (reused)", "same content as the stored version; nothing re-embedded"
                    else:
                        row["status"], row["detail"] = "updated", "content changed; stale chunks were replaced"
                rows.append(row)
            counters = live()
            job.update(
                status="done", report=report.to_dict(), file_results=rows,
                model_load_s=0.0 if was_ready else round(self.embedder.load_seconds, 2), **counters,
            )
            self._history.append({
                "run": job.label, "files": len(ready), "wall s": round(job.snapshot()["elapsed_s"], 2),
                "parsed chunks": report.chunks_generated, "embedded by model": int(counters["model_texts"]),
                "reused from cache": int(counters["cache_hits"] + counters["deduplicated"]),
                "files not parsed (unchanged)": report.documents_skipped_without_parsing,
            })
        except _Cancelled:
            job.update(status="cancelled")
        except Exception as exc:
            job.update(status="failed", error=f"{type(exc).__name__}: {exc}", error_detail=traceback.format_exc())
        finally:
            job.finish()
