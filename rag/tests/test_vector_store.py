import pytest

from src.text_chunker import Chunk
from src.vector_store import ChromaVectorStore, VectorStoreError, build_where, normalize_metadata
from tests.conftest import FakeEmbedder

EMB = FakeEmbedder()


def chunk(cid, text, project="project_alpha", path="a.txt", **meta):
    base = {"source_path": path, "source_file": path, "project_id": project, "doc_type": "txt", "page_number": 0}
    base.update(meta)
    return Chunk(id=cid, text=text, metadata=base)


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(tmp_path / "db", "unit_test", EMB.model_name, EMB.dimension)


def add(store, *chunks):
    return store.upsert_chunks(list(chunks), EMB.embed_documents([c.text for c in chunks]))


def test_upsert_is_idempotent(store):
    c = chunk("1", "resource shortage risk")
    add(store, c)
    add(store, c)
    assert store.count() == 1


def test_upsert_updates_text(store):
    add(store, chunk("1", "old text"))
    add(store, chunk("1", "new text"))
    assert store.query(EMB.embed_query("new text"), 1)[0].text == "new text"


def test_query_returns_text_metadata_distance_ordered(store):
    add(store, chunk("1", "vendor delay budget overrun"), chunk("2", "resource shortage staff attrition"), chunk("3", "regulatory approval"))
    results = store.query(EMB.embed_query("staff resource shortage"), top_k=3)
    assert results[0].id == "2"
    assert [r.distance for r in results] == sorted(r.distance for r in results)
    top = results[0]
    assert top.text and top.metadata["source_path"] == "a.txt"
    assert top.distance == pytest.approx(0.0, abs=0.5) and top.similarity == pytest.approx(1 - top.distance)
    assert "ingested_at" in top.metadata and top.metadata["embedding_model"] == EMB.model_name


def test_empty_collection_and_no_match_filter(store):
    assert store.query(EMB.embed_query("anything"), 5) == []
    add(store, chunk("1", "text"))
    assert store.query(EMB.embed_query("text"), 5, where={"project_id": "nope"}) == []


def test_metadata_filters(store):
    add(store, chunk("1", "risk alpha", project="project_alpha"), chunk("2", "risk beta", project="project_beta", doc_type="csv"))
    q = EMB.embed_query("risk")
    assert {r.id for r in store.query(q, 5, build_where(project_id="project_alpha"))} == {"1"}
    assert {r.id for r in store.query(q, 5, build_where(project_ids=["project_alpha", "project_beta"]))} == {"1", "2"}
    assert {r.id for r in store.query(q, 5, build_where(project_ids=["project_beta"], doc_types=["csv"]))} == {"2"}
    assert store.query(q, 5, build_where(project_ids=["project_beta"], doc_types=["txt"])) == []


def test_build_where_shapes():
    assert build_where() is None
    assert build_where(project_id="p") == {"project_id": "p"}
    assert "$and" in build_where(project_id="p", doc_types=["pdf"])


def test_persistence_across_reinitialisation(tmp_path):
    first = ChromaVectorStore(tmp_path / "db", "persist_test", EMB.model_name, EMB.dimension)
    add(first, chunk("1", "persisted risk text"))
    del first
    second = ChromaVectorStore(tmp_path / "db", "persist_test", EMB.model_name, EMB.dimension)
    assert second.count() == 1
    assert second.query(EMB.embed_query("persisted risk"), 1)[0].id == "1"


def test_reopen_without_model_learns_dimension(tmp_path):
    ChromaVectorStore(tmp_path / "db", "learn_test", EMB.model_name, EMB.dimension)
    again = ChromaVectorStore(tmp_path / "db", "learn_test")
    assert again.embedding_dim == EMB.dimension and again.embedding_model == EMB.model_name


def test_dimension_and_model_mismatch_rejected(tmp_path):
    ChromaVectorStore(tmp_path / "db", "guard_test", "model-a", 64)
    with pytest.raises(VectorStoreError, match="dimensional"):
        ChromaVectorStore(tmp_path / "db", "guard_test", "model-a", 128)
    with pytest.raises(VectorStoreError, match="embedding model"):
        ChromaVectorStore(tmp_path / "db", "guard_test", "model-b", 64)


def test_wrong_vector_sizes_rejected(store):
    with pytest.raises(VectorStoreError):
        store.upsert_chunks([chunk("1", "t")], [[0.1, 0.2]])
    with pytest.raises(VectorStoreError):
        store.upsert_chunks([chunk("1", "t")], [])
    with pytest.raises(VectorStoreError):
        store.query([0.1, 0.2], 1)


def test_delete_source_keeps_selected_ids(store):
    add(store, chunk("1", "old a", path="x.txt"), chunk("2", "old b", path="x.txt"), chunk("3", "other", path="y.txt"))
    assert store.delete_source("x.txt", keep_ids={"1"}) == 1
    assert store.source_chunk_ids("x.txt") == {"1"} and store.count() == 2


def test_stats_and_sources(store):
    add(store, chunk("1", "a", path="x.txt"), chunk("2", "b", path="x.txt", project="project_beta"), chunk("3", "c", path="y.txt"))
    stats = store.stats()
    assert stats["total_chunks"] == 3 and stats["total_sources"] == 2
    assert stats["chunks_per_project"] == {"project_alpha": 2, "project_beta": 1}
    assert store.list_sources() == {"x.txt": 2, "y.txt": 1}


def test_reset_drops_collection_but_keeps_directory(store, tmp_path):
    add(store, chunk("1", "text"))
    store.reset()
    assert store.count() == 0 and (tmp_path / "db").is_dir()
    add(store, chunk("2", "again"))  # still usable
    assert store.count() == 1


def test_normalize_metadata():
    meta = normalize_metadata({"a": None, "b": [1, 2], "c": True, "d": 3, "e": 1.5, "f": object, "g": "x"})
    assert "a" not in meta and meta["b"] == "1, 2" and meta["c"] is True and meta["d"] == 3 and meta["g"] == "x"
    assert isinstance(meta["f"], str)
