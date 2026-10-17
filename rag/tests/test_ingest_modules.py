"""Manifest, worker, model-keyed vector cache and upload preparation. No model download, no ChromaDB needed."""
from pathlib import Path

import pytest

import src.embedding_cache as ec
from src.config import RAGSettings
from src.embedding_cache import EmbeddingCache, ManagedEmbedder
from src.ingest_job import prepare_uploads
from src.ingest_manifest import STATUS_OK, IngestManifest, ManifestEntry, compute_signature, manifest_path
from src.ingest_worker import prepare_file
from src.text_chunker import ChunkingConfig


class Upload:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


class Inner:
    dimension = 2
    calls = 0

    def embed_documents(self, texts):
        Inner.calls += 1
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 1.0]


def test_manifest_roundtrip_and_model_change_ignores_entries(tmp_path):
    s = RAGSettings.from_env()
    path, sig = manifest_path(tmp_path, "c1"), compute_signature(s, "m1")
    m = IngestManifest.load(path, sig)
    m.record("a.txt", ManifestEntry("h", STATUS_OK, 3))
    m.save()
    assert IngestManifest.load(path, sig).get("a.txt").chunks == 3
    assert IngestManifest.load(path, compute_signature(s, "m2")).entries == {}
    assert IngestManifest.load(path, sig).indexed_hashes() == {"h": "a.txt"}


def test_prepare_file_ok_empty_failed(tmp_path):
    (tmp_path / "x.txt").write_text("Project Alpha is delayed. The vendor was late.")
    (tmp_path / "e.txt").write_text("   ")
    (tmp_path / "b.pdf").write_bytes(b"nope")
    cfg = ChunkingConfig()
    assert prepare_file(str(tmp_path / "x.txt"), str(tmp_path), "x.txt", 2, 100, cfg).chunks
    assert prepare_file(str(tmp_path / "e.txt"), str(tmp_path), "e.txt", 2, 100, cfg).status == "empty"
    assert prepare_file(str(tmp_path / "b.pdf"), str(tmp_path), "b.pdf", 2, 100, cfg).status == "failed"


def test_cache_is_keyed_by_model(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    cache.put_many("bge", ["t"], [[1.0, 2.0]])
    assert cache.get_many("bge", ["t"])[0] == [1.0, 2.0]
    assert cache.get_many("minilm", ["t"]) == {}  # a different model never receives old vectors


def test_managed_embedder_batches_dedupes_caches_and_loads_once(tmp_path, monkeypatch):
    Inner.calls = 0
    monkeypatch.setattr(ec, "create_embedder", lambda settings: Inner())
    emb = ManagedEmbedder(RAGSettings.from_env(), cache=EmbeddingCache(tmp_path / "c.sqlite"))
    assert emb.embed_documents(["aa", "aa", "bbb"]) == [[2.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert emb.model_texts == 2 and emb.deduplicated == 1
    emb.embed_documents(["aa", "bbb"])
    assert emb.cache_hits == 2 and Inner.calls == 1 and emb.ready


def test_prepare_uploads_validates_and_detects_duplicates(tmp_path):
    docs = tmp_path / "documents"
    items = prepare_uploads(
        [Upload("a.txt", b"hello"), Upload("copy.txt", b"hello"), Upload("x.zip", b"PK"), Upload("z.txt", b"")],
        docs / "uploads" / "p", docs,
    )
    assert [i.status for i in items] == ["ready", "duplicate (not indexed)", "unsupported", "empty"]
    assert items[0].relative == "uploads/p/a.txt" and not (docs / "uploads/p/copy.txt").exists()
    again = prepare_uploads([Upload("b.txt", b"hello")], docs / "uploads" / "p", docs, known_hashes={items[0].sha256: "uploads/p/a.txt"})
    assert again[0].status == "duplicate (not indexed)" and not (docs / "uploads/p/b.txt").exists()
