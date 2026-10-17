"""Text cleaning and chunking.

Design
------
* **Prose (PDF / DOCX / TXT)** is chunked *within* a page or heading section, so
  every chunk maps to exactly one page number / section and citations stay
  accurate. Chunks are built from whole paragraphs, then sentences, then words
  (only as a last resort), with a configurable overlap.
* **CSV (e.g. risk registers)** is chunked by *packing whole rows*. A row is
  never split unless a single row alone exceeds the chunk size. Every chunk
  starts with a header line naming the file, the project and **all column
  names**, and neighbouring rows are packed together (plus ``csv_overlap_rows``
  rows of overlap) so a risk description stays next to its mitigation fields
  and adjacent records. Rows of different projects are never mixed in a chunk.
* Chunk sizes are *approximate tokens* (see :func:`estimate_tokens`). The
  default (350) stays under the 512-token limit of the default embedding model.

Chunk ids are deterministic: ``<hash of path>-<index>-<hash of text>``. The same
file content always yields the same ids, so re-ingestion upserts rather than
duplicates, and edited content produces new ids.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .document_loader import DocumentSection, LoadedDocument

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MIN_BUDGET_TOKENS = 50


def estimate_tokens(text: str) -> int:
    """Cheap, conservative token estimate (no tokenizer dependency).

    ``max(words * 1.3, chars / 4)`` slightly over-estimates English prose, which
    keeps chunks safely below the embedding model's sequence limit.
    """
    if not text:
        return 0
    return max(math.ceil(len(text.split()) * 1.3), math.ceil(len(text) / 4))


def clean_text(text: str) -> str:
    """Normalise extracted text without destroying line structure (needed for CSV rows)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50
    csv_overlap_rows: int = 1

    def __post_init__(self) -> None:
        if self.chunk_size_tokens < _MIN_BUDGET_TOKENS:
            raise ValueError(f"chunk_size_tokens must be >= {_MIN_BUDGET_TOKENS}")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be >= 0 and smaller than chunk_size_tokens")
        if self.csv_overlap_rows < 0:
            raise ValueError("csv_overlap_rows must be >= 0")


@dataclass
class Chunk:
    """A chunk of text ready for embedding. ``metadata`` values are all scalars."""

    id: str
    text: str
    metadata: Dict[str, Any]


def make_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    path_hash = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:10]
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{path_hash}-{chunk_index:05d}-{text_hash}"


# --------------------------------------------------------------------------- prose
Unit = Tuple[str, bool]  # (text, starts_new_paragraph)


def _split_long_text_by_words(text: str, max_tokens: int) -> List[str]:
    words = text.split()
    words_per_piece = max(1, int(max_tokens / 1.4))
    pieces = [" ".join(words[i : i + words_per_piece]) for i in range(0, len(words), words_per_piece)]
    # Very long "words" (e.g. base64) can still exceed the budget: hard split by characters.
    max_chars = max_tokens * 4
    result: List[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            result.append(piece[:max_chars])
            piece = piece[max_chars:]
        if piece:
            result.append(piece)
    return result


def _to_units(text: str, max_tokens: int) -> List[Unit]:
    units: List[Unit] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if estimate_tokens(paragraph) <= max_tokens:
            units.append((paragraph, True))
            continue
        first = True
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if estimate_tokens(sentence) <= max_tokens:
                units.append((sentence, first))
                first = False
            else:
                for piece in _split_long_text_by_words(sentence, max_tokens):
                    units.append((piece, first))
                    first = False
    return units


def _join_units(units: List[Unit]) -> str:
    out = ""
    for index, (text, new_paragraph) in enumerate(units):
        if index == 0:
            out = text
        else:
            out += ("\n\n" if new_paragraph else " ") + text
    return out


def split_prose(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Split prose into chunks of at most ``max_tokens`` with ``overlap_tokens`` of trailing overlap."""
    units = _to_units(text, max_tokens)
    chunks: List[str] = []
    current: List[Unit] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit[0])
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append(_join_units(current))
            overlap: List[Unit] = []
            overlap_total = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous[0])
                if overlap_total + previous_tokens > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_total += previous_tokens
            while overlap and overlap_total + unit_tokens > max_tokens:
                overlap_total -= estimate_tokens(overlap.pop(0)[0])
            current, current_tokens = overlap, overlap_total
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append(_join_units(current))
    return chunks


# --------------------------------------------------------------------------- CSV
@dataclass
class _RowItem:
    row_number: int
    text: str
    tokens: int


def _csv_prefix(doc: LoadedDocument, project_id: str, max_tokens: int) -> str:
    columns = list(doc.columns)
    while True:
        shown = ", ".join(columns) + (f", ... (+{len(doc.columns) - len(columns)} more)" if len(columns) < len(doc.columns) else "")
        prefix = f"[CSV file: {doc.source_file} | Project: {project_id} | Columns: {shown}]\n"
        if estimate_tokens(prefix) <= max_tokens * 0.4 or len(columns) <= 1:
            return prefix
        columns = columns[: max(1, len(columns) - max(1, len(columns) // 5))]


def _row_items(sections: List[DocumentSection], budget: int) -> List[_RowItem]:
    items: List[_RowItem] = []
    for section in sections:
        text = clean_text(section.text)
        if not text:
            continue
        tokens = estimate_tokens(text)
        if tokens <= budget:
            items.append(_RowItem(section.row_number, text, tokens))
            continue
        # A single oversized row: split on field lines, repeating the row label.
        label = f"Row {section.row_number}"
        lines = text.split("\n")[1:]  # drop the original "Row N:" line
        parts: List[str] = []
        buffer: List[str] = []
        for line in lines:
            for piece in ([line] if estimate_tokens(line) <= budget - 10 else _split_long_text_by_words(line, budget - 10)):
                candidate = "\n".join(buffer + [piece])
                if buffer and estimate_tokens(candidate) > budget - 10:
                    parts.append("\n".join(buffer))
                    buffer = []
                buffer.append(piece)
        if buffer:
            parts.append("\n".join(buffer))
        total = len(parts)
        for number, part in enumerate(parts, start=1):
            part_text = f"{label} (part {number}/{total}):\n{part}"
            items.append(_RowItem(section.row_number, part_text, estimate_tokens(part_text)))
    return items


def _pack_rows(items: List[_RowItem], budget: int, overlap_rows: int) -> List[List[_RowItem]]:
    groups: List[List[_RowItem]] = []
    current: List[_RowItem] = []
    current_tokens = 0
    for item in items:
        if current and current_tokens + item.tokens > budget:
            groups.append(current)
            overlap: List[_RowItem] = []
            if overlap_rows > 0:
                wanted: List[int] = []
                for previous in reversed(current):
                    if previous.row_number not in wanted:
                        if len(wanted) == overlap_rows:
                            break
                        wanted.append(previous.row_number)
                overlap = [p for p in current if p.row_number in wanted and p.row_number != item.row_number]
            overlap_tokens = sum(p.tokens for p in overlap)
            while overlap and overlap_tokens + item.tokens > budget:
                overlap_tokens -= overlap.pop(0).tokens
            current, current_tokens = overlap, overlap_tokens
        current.append(item)
        current_tokens += item.tokens
    if current:
        groups.append(current)
    return groups


def _chunk_csv(doc: LoadedDocument, config: ChunkingConfig) -> List[Tuple[str, Dict[str, Any]]]:
    """Return ``(text, extra_metadata)`` pairs for a CSV document."""
    # Group by project across the WHOLE file (not only consecutive runs): an unsorted
    # register (A, B, A, B ...) would otherwise produce one tiny chunk per row.
    grouped: Dict[str, List[DocumentSection]] = {}
    for section in doc.sections:
        grouped.setdefault(section.project_id or doc.project_id, []).append(section)
    by_project = list(grouped.items())

    results: List[Tuple[str, Dict[str, Any]]] = []
    for project, sections in by_project:
        prefix = _csv_prefix(doc, project, config.chunk_size_tokens)
        budget = max(_MIN_BUDGET_TOKENS, config.chunk_size_tokens - estimate_tokens(prefix))
        items = _row_items(sections, budget)
        for group in _pack_rows(items, budget, config.csv_overlap_rows):
            body = "\n\n".join(item.text for item in group)
            rows = [item.row_number for item in group]
            results.append(
                (prefix + body, {"project_id": project, "row_start": min(rows), "row_end": max(rows)})
            )
    return results


# --------------------------------------------------------------------------- public API
def chunk_document(doc: LoadedDocument, config: ChunkingConfig | None = None) -> List[Chunk]:
    """Chunk one :class:`LoadedDocument`. Returns ``[]`` if it has no usable text."""
    config = config or ChunkingConfig()
    pieces: List[Tuple[str, Dict[str, Any]]] = []

    if doc.doc_type == "csv":
        pieces = _chunk_csv(doc, config)
    else:
        for section in doc.sections:
            text = clean_text(section.text)
            if not text:
                continue
            prefix = f"[Section: {section.section}]\n" if section.section else ""
            budget = max(_MIN_BUDGET_TOKENS, config.chunk_size_tokens - estimate_tokens(prefix))
            overlap = min(config.chunk_overlap_tokens, budget // 2)
            extra = {"page_number": section.page_number, "section": section.section}
            for piece in split_prose(text, budget, overlap):
                pieces.append((prefix + piece, extra))

    chunks: List[Chunk] = []
    total = len(pieces)
    for index, (text, extra) in enumerate(pieces):
        if not text.strip():
            continue
        metadata: Dict[str, Any] = {
            "source_file": doc.source_file,
            "source_path": doc.source_path,
            "doc_type": doc.doc_type,
            "project_id": doc.project_id,
            "page_number": 0,
            "section": "",
            "row_start": 0,
            "row_end": 0,
            "chunk_index": index,
            "chunk_count": total,
            "doc_hash": doc.doc_hash,
        }
        metadata.update(extra)
        chunks.append(Chunk(id=make_chunk_id(doc.source_path, index, text), text=text, metadata=metadata))
    return chunks