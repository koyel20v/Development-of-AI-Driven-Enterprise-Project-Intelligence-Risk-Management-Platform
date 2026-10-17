"""Ingestion manifest: remembers the SHA-256 of every file that was indexed, so unchanged files are skipped
without being parsed again.

The manifest is stored next to the ChromaDB data (one JSON file per collection). It is tied to a *signature*
of the ingestion settings (embedding model, chunk size, overlap, project level). If any of these change, the
stored entries are ignored (never deleted from ChromaDB) and files are re-checked; chunks whose ids did not
change are still not re-embedded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_EMPTY = "empty"  # readable file without extractable text (e.g. a scanned PDF): not parsed again
_VERSION = 1


@dataclass
class ManifestEntry:
    sha256: str
    status: str
    chunks: int


def manifest_path(chroma_dir: Path | str, collection_name: str) -> Path:
    return Path(chroma_dir) / f"ingest_manifest_{collection_name}.json"


def compute_signature(settings, model_name: str) -> str:
    payload = json.dumps(
        {
            "v": _VERSION,
            "model": model_name,
            "chunk": settings.chunk_size_tokens,
            "overlap": settings.chunk_overlap_tokens,
            "csv_overlap": settings.csv_overlap_rows,
            "project_level": settings.project_folder_level,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class IngestManifest:
    def __init__(self, path: Path | str, signature: str) -> None:
        self.path = Path(path)
        self.signature = signature
        self.entries: Dict[str, ManifestEntry] = {}
        self.note = ""

    @classmethod
    def load(cls, path: Path | str, signature: str) -> "IngestManifest":
        manifest = cls(path, signature)
        if not manifest.path.is_file():
            manifest.note = "new manifest"
            return manifest
        try:
            data = json.loads(manifest.path.read_text(encoding="utf-8"))
            if data.get("signature") != signature:
                manifest.note = "ignored (embedding model or chunking settings changed)"
                return manifest
            for relative, raw in (data.get("files") or {}).items():
                manifest.entries[relative] = ManifestEntry(str(raw["sha256"]), str(raw["status"]), int(raw["chunks"]))
            manifest.note = f"{len(manifest.entries)} known files"
        except Exception as exc:  # corrupt file: start over, never crash ingestion
            logger.warning("Ignoring unreadable manifest %s: %s", manifest.path, exc)
            manifest.entries.clear()
            manifest.note = "ignored (unreadable)"
        return manifest

    def get(self, relative: str) -> Optional[ManifestEntry]:
        return self.entries.get(relative)

    def record(self, relative: str, entry: ManifestEntry) -> None:
        self.entries[relative] = entry

    def remove(self, relative: str) -> None:
        self.entries.pop(relative, None)

    def indexed_hashes(self) -> Dict[str, str]:
        """``sha256 -> relative path`` of every successfully indexed file (used for duplicate detection)."""
        return {e.sha256: rel for rel, e in self.entries.items() if e.status == STATUS_OK}

    def save(self) -> None:
        payload = {
            "signature": self.signature,
            "files": {rel: {"sha256": e.sha256, "status": e.status, "chunks": e.chunks} for rel, e in self.entries.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: an interrupted run never leaves a half-written manifest

    def delete_file(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
