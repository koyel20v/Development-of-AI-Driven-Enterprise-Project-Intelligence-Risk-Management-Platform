"""Document discovery and text extraction.

Responsibilities
----------------
* Recursively scan a documents directory.
* Detect file types by extension (PDF, DOCX, TXT/MD, CSV).
* Extract text into a uniform structure (:class:`LoadedDocument` made of
  :class:`DocumentSection` objects) that :mod:`text_chunker` consumes.
* Never silently drop a file: every skipped / empty / failed file is logged and
  counted in :class:`LoadStats`.

Section semantics by file type
------------------------------
* PDF  - one section per page (``page_number`` is 1-based).
* DOCX - one section per heading block (``section`` holds the heading text);
         tables are rendered as ``Header: value`` lines.
* TXT  - a single section for the whole file.
* CSV  - one section per data row (``row_number`` is 1-based, header excluded;
         spreadsheet line number = ``row_number + 1``). ``LoadedDocument.columns``
         keeps the header so the chunker can repeat it in every chunk.

Project identifier
------------------
Determined, in priority order, from: a CSV column named ``project_id`` /
``project`` / ``project_name`` (row level), then the folder layout
``documents/<type-folder>/<project-folder>/file`` (controlled by
``project_folder_level``), otherwise ``"unassigned"``.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "txt",
    ".csv": "csv",
}
UNASSIGNED_PROJECT = "unassigned"
DEFAULT_MAX_FILE_MB = 100
_PROJECT_COLUMN_NAMES = {"project_id", "project", "project_name", "project name", "projectid"}


class DocumentLoadError(Exception):
    """A file could not be read (corrupted, encrypted, binary, too large, ...)."""


class EmptyDocumentError(Exception):
    """A file was readable but contained no extractable text."""


@dataclass
class DocumentSection:
    """A unit of extracted text with its location inside the source file."""

    text: str
    page_number: int = 0  # PDF page (1-based); 0 = not applicable
    section: str = ""  # DOCX heading text; "" = not applicable
    row_number: int = 0  # CSV data row (1-based); 0 = not applicable
    project_id: str = ""  # row-level project override (CSV); "" = use document project


@dataclass
class LoadedDocument:
    source_file: str
    source_path: str  # POSIX path relative to the documents root
    doc_type: str  # pdf | docx | txt | csv
    project_id: str
    doc_hash: str  # sha256 of file bytes
    sections: List[DocumentSection]
    columns: List[str] = field(default_factory=list)  # CSV header

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.sections)


@dataclass
class LoadStats:
    """Ingestion statistics. ``failures`` and the file lists contain relative paths."""

    files_found: int = 0
    documents_loaded: int = 0
    sections_loaded: int = 0
    pages_failed: int = 0
    unsupported_files: List[str] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)
    failures: List[Tuple[str, str]] = field(default_factory=list)
    discovered_paths: List[str] = field(default_factory=list)  # every supported file seen

    @property
    def files_skipped_unsupported(self) -> int:
        return len(self.unsupported_files)

    @property
    def files_empty(self) -> int:
        return len(self.empty_files)

    @property
    def files_failed(self) -> int:
        return len(self.failures)

    def summary(self) -> Dict[str, int]:
        return {
            "files_found": self.files_found,
            "documents_loaded": self.documents_loaded,
            "sections_loaded": self.sections_loaded,
            "unsupported": self.files_skipped_unsupported,
            "empty": self.files_empty,
            "failed": self.files_failed,
            "pdf_pages_failed": self.pages_failed,
        }


# --------------------------------------------------------------------------- helpers
def slugify_project_id(value: str) -> str:
    """Normalise a project name to a filter-friendly id: ``"Project Alpha"`` -> ``"project_alpha"``."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or UNASSIGNED_PROJECT


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_bytes(raw: bytes) -> str:
    if b"\x00" in raw[:4096]:
        raise DocumentLoadError("file looks binary (contains NUL bytes)")
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("could not decode file text")  # pragma: no cover (latin-1 never fails)


def project_from_path(relative: Path, project_folder_level: int) -> str:
    """Derive project id from folder layout. Level 2 => ``<type>/<project>/file``."""
    if project_folder_level <= 0:
        return UNASSIGNED_PROJECT
    directories = relative.parts[:-1]
    if len(directories) >= project_folder_level:
        return slugify_project_id(directories[project_folder_level - 1])
    return UNASSIGNED_PROJECT


# --------------------------------------------------------------------------- extractors
def _extract_pdf(path: Path, stats: LoadStats) -> List[DocumentSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError("pypdf is not installed (pip install pypdf)") from exc

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise DocumentLoadError(f"encrypted PDF could not be opened: {exc}") from exc
            if not decrypted:
                raise DocumentLoadError("PDF is password protected")
        pages = list(enumerate(reader.pages, start=1))
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"corrupted or unreadable PDF: {exc}") from exc

    sections: List[DocumentSection] = []
    for page_number, page in pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            stats.pages_failed += 1
            logger.warning("%s: page %d could not be extracted (%s)", path.name, page_number, exc)
            continue
        if text.strip():
            sections.append(DocumentSection(text=text, page_number=page_number))
    return sections


def _docx_table_to_text(table) -> str:
    rows: List[List[str]] = []
    for row in table.rows:
        cells: List[str] = []
        previous: Optional[str] = None
        for cell in row.cells:
            value = " ".join(cell.text.split())
            if value == previous:  # merged cells repeat their text
                continue
            cells.append(value)
            previous = value
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    if len(rows) == 1:
        return " | ".join(rows[0])
    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        pairs = [
            f"{header[i] if i < len(header) and header[i] else f'column_{i + 1}'}: {value}"
            for i, value in enumerate(row)
            if value
        ]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _extract_docx(path: Path) -> List[DocumentSection]:
    try:
        import docx  # python-docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError("python-docx is not installed (pip install python-docx)") from exc

    try:
        document = docx.Document(str(path))
        blocks = list(document.iter_inner_content())
    except Exception as exc:
        raise DocumentLoadError(f"corrupted or unreadable DOCX: {exc}") from exc

    sections: List[DocumentSection] = []
    heading = ""
    buffer: List[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(DocumentSection(text=text, section=heading))
        buffer.clear()

    for block in blocks:
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name if block.style is not None else "") or ""
            if style.startswith("Heading") or style == "Title":
                flush()
                heading = text
            buffer.append(text)
        elif isinstance(block, Table):
            table_text = _docx_table_to_text(block)
            if table_text:
                buffer.append(table_text)
    flush()
    return sections


def _extract_txt(path: Path) -> List[DocumentSection]:
    try:
        text = _decode_bytes(path.read_bytes())
    except OSError as exc:
        raise DocumentLoadError(f"cannot read file: {exc}") from exc
    return [DocumentSection(text=text)] if text.strip() else []


def _dedupe_header(raw_header: List[str]) -> List[str]:
    header: List[str] = []
    seen: Dict[str, int] = {}
    for index, name in enumerate(raw_header, start=1):
        name = " ".join(name.split()) or f"column_{index}"
        count = seen.get(name.lower(), 0)
        seen[name.lower()] = count + 1
        header.append(name if count == 0 else f"{name}_{count + 1}")
    return header


def _extract_csv(path: Path, doc_project: str) -> Tuple[List[DocumentSection], List[str]]:
    try:
        text = _decode_bytes(path.read_bytes())
    except OSError as exc:
        raise DocumentLoadError(f"cannot read file: {exc}") from exc
    if not text.strip():
        return [], []

    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    csv.field_size_limit(10_000_000)

    try:
        reader = csv.reader(text.splitlines(), dialect)
        header: Optional[List[str]] = None
        sections: List[DocumentSection] = []
        row_number = 0
        for cells in reader:
            if not any(cell.strip() for cell in cells):
                continue
            if header is None:
                header = _dedupe_header(cells)
                continue
            row_number += 1
            pairs: List[Tuple[str, str]] = []
            row_project = ""
            for index, cell in enumerate(cells):
                value = " ".join(cell.split())
                if not value:
                    continue
                name = header[index] if index < len(header) else f"extra_{index + 1}"
                pairs.append((name, value))
                if name.lower() in _PROJECT_COLUMN_NAMES and not row_project:
                    row_project = slugify_project_id(value)
            if not pairs:
                continue
            body = "\n".join(f"{name}: {value}" for name, value in pairs)
            sections.append(
                DocumentSection(
                    text=f"Row {row_number}:\n{body}",
                    row_number=row_number,
                    project_id=row_project or doc_project,
                )
            )
    except csv.Error as exc:
        raise DocumentLoadError(f"malformed CSV: {exc}") from exc
    return sections, (header or [])


# --------------------------------------------------------------------------- public API
def load_file(
    path: Path,
    root: Path,
    stats: Optional[LoadStats] = None,
    project_folder_level: int = 2,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
) -> LoadedDocument:
    """Load one file.

    Raises:
        DocumentLoadError: unsupported/corrupted/too large/binary file.
        EmptyDocumentError: file readable but no text extracted.
    """
    stats = stats if stats is not None else LoadStats()
    path, root = Path(path).resolve(), Path(root).resolve()
    extension = path.suffix.lower()
    doc_type = SUPPORTED_EXTENSIONS.get(extension)
    if doc_type is None:
        raise DocumentLoadError(f"unsupported file type {extension or '(none)'}")
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError as exc:
        raise DocumentLoadError(f"cannot stat file: {exc}") from exc
    if size_mb > max_file_mb:
        raise DocumentLoadError(f"file is {size_mb:.0f} MB, above the {max_file_mb} MB limit")

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    project_id = project_from_path(relative, project_folder_level)

    columns: List[str] = []
    if doc_type == "pdf":
        sections = _extract_pdf(path, stats)
    elif doc_type == "docx":
        sections = _extract_docx(path)
    elif doc_type == "csv":
        sections, columns = _extract_csv(path, project_id)
    else:
        sections = _extract_txt(path)

    if not sections:
        hint = " (scanned/image-only PDFs need OCR, which is not supported)" if doc_type == "pdf" else ""
        raise EmptyDocumentError(f"no extractable text{hint}")

    try:
        doc_hash = _file_sha256(path)
    except OSError as exc:
        raise DocumentLoadError(f"cannot read file: {exc}") from exc

    return LoadedDocument(
        source_file=path.name,
        source_path=relative.as_posix(),
        doc_type=doc_type,
        project_id=project_id,
        doc_hash=doc_hash,
        sections=sections,
        columns=columns,
    )


def _iter_files(root: Path, recursive: bool) -> Iterator[Path]:
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        for name in sorted(files):
            if name.startswith(".") or name.startswith("~$"):  # hidden files, Office lock files
                continue
            yield Path(current) / name
        if not recursive:
            break


def iter_documents(
    root: Path,
    stats: Optional[LoadStats] = None,
    recursive: bool = True,
    project_folder_level: int = 2,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
) -> Iterator[LoadedDocument]:
    """Yield every loadable document under ``root``; problems are logged and recorded in ``stats``."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {root}")
    stats = stats if stats is not None else LoadStats()

    for path in _iter_files(root, recursive):
        stats.files_found += 1
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            stats.unsupported_files.append(relative)
            logger.warning("Skipping unsupported file type: %s", relative)
            continue
        stats.discovered_paths.append(relative)
        try:
            document = load_file(path, root, stats, project_folder_level, max_file_mb)
        except EmptyDocumentError as exc:
            stats.empty_files.append(relative)
            logger.warning("Empty document %s: %s", relative, exc)
            continue
        except DocumentLoadError as exc:
            stats.failures.append((relative, str(exc)))
            logger.warning("Failed to load %s: %s", relative, exc)
            continue
        except Exception as exc:  # defensive: one bad file must never stop the run
            stats.failures.append((relative, f"unexpected error: {exc}"))
            logger.exception("Unexpected error loading %s", relative)
            continue
        stats.documents_loaded += 1
        stats.sections_loaded += len(document.sections)
        yield document


def load_directory(
    root: Path,
    recursive: bool = True,
    project_folder_level: int = 2,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
) -> Tuple[List[LoadedDocument], LoadStats]:
    """Convenience wrapper that loads everything into memory and returns ``(documents, stats)``."""
    stats = LoadStats()
    documents = list(iter_documents(root, stats, recursive, project_folder_level, max_file_mb))
    return documents, stats
