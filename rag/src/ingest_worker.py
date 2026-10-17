"""Parse + chunk ONE file. Top-level and picklable so it can run in a process pool."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .document_loader import DocumentLoadError, EmptyDocumentError, LoadStats, load_file
from .text_chunker import Chunk, ChunkingConfig, chunk_document


@dataclass
class PreparedFile:
    status: str = "ok"  # "ok" | "empty" | "failed"
    error: str = ""
    chunks: List[Chunk] = field(default_factory=list)
    sections: int = 0
    load_s: float = 0.0
    chunk_s: float = 0.0
    pages_failed: int = 0


def prepare_file(path: str, root: str, relative: str, level: int, max_mb: int, config: ChunkingConfig) -> PreparedFile:
    stats = LoadStats()
    started = time.perf_counter()
    try:
        doc = load_file(Path(path), Path(root), stats, level, max_mb)
    except EmptyDocumentError as exc:
        return PreparedFile("empty", str(exc), load_s=time.perf_counter() - started, pages_failed=stats.pages_failed)
    except DocumentLoadError as exc:
        return PreparedFile("failed", str(exc), load_s=time.perf_counter() - started, pages_failed=stats.pages_failed)
    except Exception as exc:  # never let one bad file stop a batch
        return PreparedFile("failed", f"{type(exc).__name__}: {exc}", load_s=time.perf_counter() - started)
    loaded = time.perf_counter()
    chunks = chunk_document(doc, config)
    chunked = time.perf_counter()
    if not chunks:
        return PreparedFile("empty", "no usable text", sections=len(doc.sections), load_s=loaded - started,
                            chunk_s=chunked - loaded, pages_failed=stats.pages_failed)
    return PreparedFile("ok", "", chunks, len(doc.sections), loaded - started, chunked - loaded, stats.pages_failed)
