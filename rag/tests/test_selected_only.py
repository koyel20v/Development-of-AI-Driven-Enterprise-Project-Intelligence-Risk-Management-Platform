"""Selected-files-only ingestion: historical files in the documents folder must never be indexed by accident."""
from src.rag_pipeline import RAGPipeline
from tests.conftest import FakeEmbedder, FakeGenerator


def make(settings, emb=None):
    return RAGPipeline(settings, embedder=emb or FakeEmbedder(), generator=FakeGenerator())


def test_only_selected_file_is_indexed(settings):
    emb = FakeEmbedder()
    pipe = make(settings, emb)
    report = pipe.ingest(only_paths=["text/project_alpha/status.txt"])
    assert report.documents_indexed == 1 and report.files_discovered == 1
    assert set(pipe.store.list_sources()) == {"text/project_alpha/status.txt"}
    assert "selected files only" in report.mode


def test_restricted_run_never_prunes_or_touches_other_documents(settings, docs_dir):
    pipe = make(settings)
    pipe.ingest()  # index everything once
    before = pipe.store.list_sources()
    (docs_dir / "text" / "project_alpha" / "status.txt").unlink()
    report = pipe.ingest(only_paths=["csv/risk_register.csv"], prune=True)  # prune must be ignored
    assert report.pruned_chunks == 0 and pipe.store.list_sources() == before


def test_reuploading_same_selected_file_embeds_nothing(settings):
    emb = FakeEmbedder()
    pipe = make(settings, emb)
    pipe.ingest(only_paths=["text/project_alpha/status.txt"])
    embedded = emb.texts_embedded
    second = pipe.ingest(only_paths=["text/project_alpha/status.txt"])
    assert second.documents_unchanged == 1 and second.chunks_embedded == 0 and emb.texts_embedded == embedded


def test_unknown_selected_path_indexes_nothing(settings):
    pipe = make(settings)
    report = pipe.ingest(only_paths=["does/not/exist.pdf"])
    assert report.documents_indexed == 0 and pipe.store.count() == 0
