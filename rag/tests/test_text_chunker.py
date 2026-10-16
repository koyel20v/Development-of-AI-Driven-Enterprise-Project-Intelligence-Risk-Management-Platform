import pytest

from src.document_loader import DocumentSection, LoadedDocument, load_directory
from src.text_chunker import ChunkingConfig, chunk_document, clean_text, estimate_tokens, make_chunk_id, split_prose


def make_doc(sections, doc_type="txt", columns=None, project="project_alpha", path="text/a.txt"):
    return LoadedDocument(
        source_file=path.split("/")[-1], source_path=path, doc_type=doc_type,
        project_id=project, doc_hash="h" * 64, sections=sections, columns=columns or [],
    )


def test_config_validation():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size_tokens=10)
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=100)
    with pytest.raises(ValueError):
        ChunkingConfig(csv_overlap_rows=-1)


def test_clean_text():
    assert clean_text("a \u00a0 b\r\n\r\n\r\n\r\nc\x00") == "a b\n\nc"
    assert clean_text("\ufb01nal") == "final"  # ligature normalised


def test_empty_content_yields_no_chunks():
    assert chunk_document(make_doc([DocumentSection(text="  \n ")])) == []
    assert chunk_document(make_doc([])) == []


def test_chunks_respect_size_and_overlap():
    sentences = [f"Sentence number {i} describes a project risk in some detail." for i in range(80)]
    text = " ".join(sentences)
    chunks = split_prose(text, max_tokens=100, overlap_tokens=25)
    assert len(chunks) > 3
    assert all(estimate_tokens(c) <= 100 for c in chunks)
    # consecutive chunks share overlapping text
    assert any(sentences[i] in chunks[0] and sentences[i] in chunks[1] for i in range(80))
    # nothing lost: every sentence appears somewhere
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_giant_unbroken_text_is_hard_split():
    chunks = split_prose("x" * 10_000, max_tokens=100, overlap_tokens=0)
    assert all(estimate_tokens(c) <= 100 for c in chunks) and len(chunks) > 1


def test_metadata_preserved_and_pages_not_mixed():
    doc = make_doc(
        [DocumentSection(text="Page one text about delay.", page_number=1),
         DocumentSection(text="Page two text about budget.", page_number=2)],
        doc_type="pdf", path="pdfs/project_alpha/p.pdf",
    )
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=10))
    assert [c.metadata["page_number"] for c in chunks] == [1, 2]
    for i, c in enumerate(chunks):
        m = c.metadata
        assert m["source_file"] == "p.pdf" and m["source_path"] == "pdfs/project_alpha/p.pdf"
        assert m["project_id"] == "project_alpha" and m["doc_type"] == "pdf"
        assert m["chunk_index"] == i and m["chunk_count"] == 2 and m["doc_hash"] == "h" * 64
        assert all(isinstance(v, (str, int, float, bool)) for v in m.values())
    assert "budget" not in chunks[0].text


def test_docx_section_prefix():
    doc = make_doc([DocumentSection(text="Resource shortage text.", section="Risks")], doc_type="docx")
    chunk = chunk_document(doc)[0]
    assert chunk.text.startswith("[Section: Risks]") and chunk.metadata["section"] == "Risks"


def test_ids_are_stable_and_content_sensitive():
    doc = make_doc([DocumentSection(text="Alpha risk text.")])
    assert [c.id for c in chunk_document(doc)] == [c.id for c in chunk_document(doc)]
    changed = make_doc([DocumentSection(text="Alpha risk text changed.")])
    assert chunk_document(doc)[0].id != chunk_document(changed)[0].id
    assert make_chunk_id("a", 0, "t") != make_chunk_id("b", 0, "t")


def test_csv_header_in_every_chunk_rows_intact_and_projects_not_mixed(docs_dir):
    doc = next(d for d in load_directory(docs_dir)[0] if d.doc_type == "csv")
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_tokens=120, chunk_overlap_tokens=10, csv_overlap_rows=1))
    assert chunks
    for c in chunks:
        assert "Columns: Risk ID, Project, Risk Description, Probability, Mitigation Plan, Owner" in c.text
    alpha = [c for c in chunks if c.metadata["project_id"] == "project_alpha"]
    beta = [c for c in chunks if c.metadata["project_id"] == "project_beta"]
    assert alpha and beta
    assert all("Regulatory approval" not in c.text for c in alpha)
    # risk description stays together with its mitigation plan
    for c in chunks:
        if "resource shortage" in c.text:
            assert "Hire two contractors" in c.text
        if "Regulatory approval" in c.text:
            assert "Engage regulator early" in c.text
    assert beta[0].metadata["row_start"] == 3 == beta[0].metadata["row_end"]


def test_csv_neighbouring_rows_packed_and_overlap():
    rows = [DocumentSection(text=f"Row {i}:\nRisk: r{i} " + "word " * 25, row_number=i, project_id="p") for i in range(1, 9)]
    doc = make_doc(rows, doc_type="csv", columns=["Risk"], path="csv/r.csv", project="p")
    no_overlap = chunk_document(doc, ChunkingConfig(chunk_size_tokens=120, chunk_overlap_tokens=0, csv_overlap_rows=0))
    with_overlap = chunk_document(doc, ChunkingConfig(chunk_size_tokens=120, chunk_overlap_tokens=0, csv_overlap_rows=1))
    assert any(c.metadata["row_end"] > c.metadata["row_start"] for c in no_overlap)  # rows packed together
    assert len(with_overlap) >= len(no_overlap)
    assert with_overlap[1].metadata["row_start"] <= with_overlap[0].metadata["row_end"]  # shared row
    for c in no_overlap:  # every row appears whole in exactly one chunk
        assert c.text.count("Row ") == c.metadata["row_end"] - c.metadata["row_start"] + 1


def test_csv_oversized_row_is_split_with_row_label():
    big = DocumentSection(text="Row 1:\n" + "\n".join(f"Field{i}: " + "value " * 30 for i in range(10)), row_number=1, project_id="p")
    doc = make_doc([big], doc_type="csv", columns=["Field0"], path="csv/r.csv", project="p")
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_tokens=120, chunk_overlap_tokens=0, csv_overlap_rows=1))
    assert len(chunks) > 1
    assert all("Row 1 (part" in c.text for c in chunks)
    assert all(estimate_tokens(c.text) <= 130 for c in chunks)
