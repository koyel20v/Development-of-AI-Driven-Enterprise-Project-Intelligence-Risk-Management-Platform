"""Shared fixtures: deterministic fake embedder / LLM, sample-file builders, temp ChromaDB."""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path
from typing import List, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAGSettings  # noqa: E402
from src.embeddings import EmbeddingError  # noqa: E402
from src.llm import Generation, LLMError  # noqa: E402


class FakeEmbedder:
    """Hashed bag-of-words embedder: texts sharing words get similar vectors. Fully offline."""

    model_name = "fake-embedder-v1"
    dimension = 64
    max_seq_length = 512

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.document_calls = 0
        self.texts_embedded = 0

    def _vec(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.document_calls += 1
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise EmbeddingError("simulated embedding failure")
        self.texts_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


class FakeGenerator:
    """Records prompts and returns a canned answer."""

    model_name = "fake-llm"

    def __init__(self, answer: str = "The risk is staff shortage [1].", fail: bool = False, stop_reason: str = "end_turn"):
        self.answer = answer
        self.fail = fail
        self.stop_reason = stop_reason
        self.calls: list = []

    def generate(self, question, passages):
        self.calls.append((question, list(passages)))
        if self.fail:
            raise LLMError("simulated provider outage")
        return Generation(text=self.answer, model=self.model_name, stop_reason=self.stop_reason)


def write_minimal_pdf(path: Path, pages: Sequence[str]) -> None:
    """Write a valid text PDF with one page per string (no third-party PDF writer needed)."""
    objects: List[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + n * 2
    for i, text in enumerate(pages):
        content_id = 4 + i * 2
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


RISK_CSV = (
    "Risk ID,Project,Risk Description,Probability,Mitigation Plan,Owner\n"
    "R-1,Project Alpha,Key engineers may leave causing a resource shortage,High,Hire two contractors,J. Rao\n"
    "R-2,Project Alpha,Vendor delivery delay may cause budget overrun,Medium,Negotiate penalty clause,A. Sen\n"
    "R-3,Project Beta,Regulatory approval could delay launch,Low,Engage regulator early,M. Das\n"
)


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """documents/{csv,docs,pdfs,text}/<project>/ layout with one file of each type."""
    root = tmp_path / "documents"
    for sub in ("csv", "docs", "pdfs", "text"):
        (root / sub).mkdir(parents=True)
    (root / "csv" / "risk_register.csv").write_text(RISK_CSV, encoding="utf-8")
    (root / "text" / "project_alpha").mkdir()
    (root / "text" / "project_alpha" / "status.txt").write_text(
        "Project Alpha schedule is delayed by three weeks because the vendor delivered late.", encoding="utf-8"
    )
    (root / "pdfs" / "project_beta").mkdir()
    write_minimal_pdf(root / "pdfs" / "project_beta" / "plan.pdf", ["Beta launch plan depends on regulator approval", "Beta budget is reviewed monthly"])
    return root


@pytest.fixture
def settings(tmp_path: Path, docs_dir: Path) -> RAGSettings:
    return RAGSettings.from_env().with_overrides(
        documents_dir=docs_dir,
        chroma_dir=tmp_path / "chroma",
        collection_name="test_collection",
        embedding_model="fake-embedder-v1",
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
