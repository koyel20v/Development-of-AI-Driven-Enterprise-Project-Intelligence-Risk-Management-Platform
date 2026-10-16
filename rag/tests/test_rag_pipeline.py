import shutil

import pytest

from src.config import ConfigError, RAGSettings
from src.rag_pipeline import NO_EVIDENCE_ANSWER, RAGPipeline, parse_citations
from src.vector_store import ChromaVectorStore, VectorStoreError
from tests.conftest import FakeEmbedder, FakeGenerator


def make_pipeline(settings, embedder=None, generator=None):
    return RAGPipeline(settings, embedder=embedder or FakeEmbedder(), generator=generator or FakeGenerator())


# ------------------------------------------------------------------ ingestion
def test_ingest_reports_and_indexes(settings):
    pipe = make_pipeline(settings)
    report = pipe.ingest()
    assert report.documents_loaded == 3 and report.failures == []
    assert report.chunks_generated == report.chunks_embedded == report.records_upserted == report.total_chunks_in_store > 0
    assert set(pipe.store.list_sources()) == {"csv/risk_register.csv", "text/project_alpha/status.txt", "pdfs/project_beta/plan.pdf"}


def test_reingest_is_idempotent_and_skips_embedding(settings):
    emb = FakeEmbedder()
    pipe = make_pipeline(settings, emb)
    first = pipe.ingest()
    embedded = emb.texts_embedded
    second = pipe.ingest()
    assert second.documents_unchanged == 3 and second.chunks_embedded == 0
    assert emb.texts_embedded == embedded
    assert second.total_chunks_in_store == first.total_chunks_in_store


def test_force_reembeds_without_duplicates(settings):
    pipe = make_pipeline(settings)
    first = pipe.ingest()
    forced = pipe.ingest(force=True)
    assert forced.chunks_embedded == first.chunks_embedded
    assert forced.total_chunks_in_store == first.total_chunks_in_store


def test_changed_file_replaces_stale_chunks(settings, docs_dir):
    pipe = make_pipeline(settings)
    pipe.ingest()
    target = docs_dir / "text" / "project_alpha" / "status.txt"
    target.write_text("Project Alpha is now on schedule after the vendor recovered.", encoding="utf-8")
    report = pipe.ingest()
    assert report.documents_indexed == 1 and report.documents_unchanged == 2
    texts = pipe.store._collection.get(where={"source_path": "text/project_alpha/status.txt"})["documents"]
    assert len(texts) == 1 and "on schedule" in texts[0] and "delayed" not in texts[0]


def test_prune_removes_deleted_files_only_when_requested(settings, docs_dir):
    pipe = make_pipeline(settings)
    pipe.ingest()
    (docs_dir / "text" / "project_alpha" / "status.txt").unlink()
    pipe.ingest()
    assert "text/project_alpha/status.txt" in pipe.store.list_sources()  # not pruned by default
    report = pipe.ingest(prune=True)
    assert report.pruned_chunks > 0 and "text/project_alpha/status.txt" not in pipe.store.list_sources()


def test_bad_files_and_embedding_failures_do_not_crash_ingestion(settings, docs_dir):
    (docs_dir / "pdfs" / "broken.pdf").write_bytes(b"garbage")
    (docs_dir / "text" / "bomb.txt").write_text("EXPLODE this document fails to embed", encoding="utf-8")
    (docs_dir / "misc.zip").write_bytes(b"PK")
    pipe = make_pipeline(settings, FakeEmbedder(fail_on="EXPLODE"))
    report = pipe.ingest()
    failed = {p for p, _ in report.failures}
    assert failed == {"pdfs/broken.pdf", "text/bomb.txt"}
    assert report.unsupported_files == ["misc.zip"]
    assert report.documents_indexed == 3  # the healthy documents were still indexed
    assert "text/bomb.txt" not in pipe.store.list_sources()


def test_ingest_missing_directory_raises_clear_error(settings, tmp_path):
    with pytest.raises(FileNotFoundError, match="Documents directory not found"):
        make_pipeline(settings).ingest(tmp_path / "missing")


def test_changing_embedding_model_is_rejected(settings):
    make_pipeline(settings).ingest()

    class OtherEmbedder(FakeEmbedder):
        model_name = "different-model"

    with pytest.raises(VectorStoreError):
        RAGPipeline(settings, embedder=OtherEmbedder(), generator=FakeGenerator()).ingest()


def test_ingestion_never_deletes_database_directory(settings):
    pipe = make_pipeline(settings)
    pipe.ingest()
    marker = settings.chroma_dir / "keep.me"
    marker.write_text("x")
    pipe.ingest(force=True)
    pipe.ingest(prune=True)
    assert marker.exists()


def test_data_survives_new_pipeline_instance(settings):
    make_pipeline(settings).ingest()
    fresh = make_pipeline(settings)
    answer = fresh.ask("resource shortage", project_ids=["project_alpha"])
    assert answer.sources


# ------------------------------------------------------------------ retrieval / filters / access
def test_project_filter_and_access_control(settings):
    pipe = make_pipeline(settings)
    pipe.ingest()
    only_alpha = pipe.retrieve("risk delay approval", top_k=10, project_ids=["project_alpha"])
    assert only_alpha and {r.metadata["project_id"] for r in only_alpha} == {"project_alpha"}
    assert pipe.retrieve("risk", project_ids=[]) == []  # no accessible projects => nothing
    assert pipe.retrieve("risk", project_id="project_beta", project_ids=["project_alpha"]) == []  # outside allowed set
    both = pipe.retrieve("risk delay approval", top_k=10)
    assert {"project_alpha", "project_beta"} <= {r.metadata["project_id"] for r in both}
    csv_only = pipe.retrieve("risk", top_k=10, doc_types=["csv"])
    assert csv_only and {r.metadata["doc_type"] for r in csv_only} == {"csv"}


def test_max_distance_cutoff(settings):
    pipe = make_pipeline(settings)
    pipe.ingest()
    assert pipe.retrieve("zzz qqq unrelated words", max_distance=0.05) == []


def test_empty_index_returns_no_results(settings):
    assert make_pipeline(settings).retrieve("anything") == []


# ------------------------------------------------------------------ question answering
def test_ask_returns_structured_grounded_answer(settings):
    gen = FakeGenerator("Resource shortage is a documented risk [1].")
    pipe = make_pipeline(settings, generator=gen)
    pipe.ingest()
    result = pipe.ask("What does the risk register say about resource shortages?", project_ids=["project_alpha"], top_k=3)
    assert result.sufficient_evidence and result.error is None and result.model == "fake-llm"
    assert result.sources[0].citation_id == 1 and result.sources[0].cited
    assert [s.cited for s in result.sources][1:] == [False] * (len(result.sources) - 1)
    top = result.sources[0]
    assert top.source_file and top.source_path and top.project_id == "project_alpha"
    assert 0 <= top.distance and top.text and "similarity" in top.__dict__
    assert result.filters_applied["project_ids"] == ["project_alpha"]
    assert set(result.timings_ms) == {"retrieval", "generation"}
    assert result.to_dict()["answer"] == result.answer
    passages = gen.calls[0][1]
    assert all(p.metadata["project_id"] == "project_alpha" for p in passages)


def test_pdf_page_and_csv_row_references(settings):
    pipe = make_pipeline(settings)
    pipe.ingest()
    pdf = next(s for s in pipe.ask("regulator approval launch plan", top_k=6).sources if s.doc_type == "pdf")
    assert pdf.page_number in (1, 2) and f"page {pdf.page_number}" in pdf.location
    csv = next(s for s in pipe.ask("resource shortage mitigation plan", top_k=6).sources if s.doc_type == "csv")
    assert csv.row_range and "row" in csv.location


def test_no_evidence_skips_llm(settings):
    gen = FakeGenerator()
    pipe = make_pipeline(settings, generator=gen)
    pipe.ingest()
    result = pipe.ask("anything at all", project_ids=[])
    assert result.answer == NO_EVIDENCE_ANSWER and not result.sufficient_evidence and result.sources == []
    assert gen.calls == []


def test_empty_index_gives_insufficient_evidence(settings):
    result = make_pipeline(settings).ask("What are the risks?")
    assert not result.sufficient_evidence and result.answer.startswith("INSUFFICIENT EVIDENCE")


def test_model_declaring_insufficient_evidence_is_flagged(settings):
    gen = FakeGenerator("INSUFFICIENT EVIDENCE: no budget figures are documented.")
    pipe = make_pipeline(settings, generator=gen)
    pipe.ingest()
    result = pipe.ask("What is the budget of Project Alpha?")
    assert not result.sufficient_evidence and not any(s.cited for s in result.sources) and result.sources


def test_uncited_answer_gets_warning_and_invalid_citations_ignored(settings):
    pipe = make_pipeline(settings, generator=FakeGenerator("Plain claim with no citation."))
    pipe.ingest()
    assert pipe.ask("risk").warnings
    pipe2 = make_pipeline(settings, generator=FakeGenerator("Claim [99]."))
    assert pipe2.ask("risk").warnings and not any(s.cited for s in pipe2.ask("risk").sources)
    assert parse_citations("a [1] b [2, 3] c [7]", 3) == {1, 2, 3}


def test_truncated_answer_warning(settings):
    pipe = make_pipeline(settings, generator=FakeGenerator("Cut [1]", stop_reason="max_tokens"))
    pipe.ingest()
    assert any("cut off" in w for w in pipe.ask("risk").warnings)


def test_llm_failure_returns_evidence_not_exception(settings):
    pipe = make_pipeline(settings, generator=FakeGenerator(fail=True))
    pipe.ingest()
    result = pipe.ask("resource shortage")
    assert result.error and "generation failed" in result.error
    assert result.sources and not result.sufficient_evidence


def test_embedding_failure_at_query_time_is_handled(settings):
    pipe = make_pipeline(settings)
    pipe.ingest()

    def boom(text):
        from src.embeddings import EmbeddingError
        raise EmbeddingError("model gone")

    pipe._embedder.embed_query = boom
    result = pipe.ask("risk")
    assert result.error and "retrieval failed" in result.error and result.sources == []


def test_question_validation(settings):
    pipe = make_pipeline(settings)
    for bad in ("", "   ", "x" * 2001):
        with pytest.raises(ValueError):
            pipe.ask(bad)


def test_context_budget_limits_passages(settings):
    small = settings.with_overrides(max_context_tokens=200)
    gen = FakeGenerator()
    pipe = make_pipeline(small, generator=gen)
    pipe.ingest()
    pipe.ask("risk delay approval budget", top_k=10)
    assert 1 <= len(gen.calls[0][1]) < 10


# ------------------------------------------------------------------ config
def test_config_validation_and_overrides(monkeypatch, settings):
    with pytest.raises(ConfigError):
        settings.with_overrides(collection_name="x")  # too short for Chroma
    with pytest.raises(ConfigError):
        settings.with_overrides(chunk_overlap_tokens=999)
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        settings.with_overrides(llm_provider="openai")
    monkeypatch.setenv("RAG_TOP_K", "abc")
    with pytest.raises(ConfigError):
        RAGSettings.from_env()


def test_llm_settings_defaults_and_legacy_fallback(monkeypatch):
    import src.config as config

    monkeypatch.setattr(config, "_load_dotenv", lambda: None)  # ignore the real rag/.env so the test is isolated
    for name in ("LLM_PROVIDER", "GROQ_MODEL", "LLM_MAX_TOKENS", "ANTHROPIC_MAX_TOKENS", "LLM_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    s = RAGSettings.from_env()
    assert s.llm_provider == "groq" and s.groq_model == "openai/gpt-oss-120b" and s.llm_max_tokens == 1500
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "900")  # legacy name still honoured
    assert RAGSettings.from_env().llm_max_tokens == 900
    monkeypatch.setenv("LLM_MAX_TOKENS", "700")  # new name wins
    assert RAGSettings.from_env().llm_max_tokens == 700