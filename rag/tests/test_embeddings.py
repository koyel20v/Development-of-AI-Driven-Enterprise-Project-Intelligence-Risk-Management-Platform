import sys
import types

import numpy as np
import pytest

from src.embeddings import BGE_QUERY_PREFIX, EmbeddingError, SentenceTransformerEmbedder, validate_embeddings


class StubModel:
    max_seq_length = 512

    def __init__(self, name, device=None):
        self.name, self.encoded, self.fail_times = name, [], 0

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("out of memory")
        self.encoded.append(list(texts))
        return np.ones((len(texts), 4), dtype="float32")


@pytest.fixture
def stub(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_validate_embeddings():
    validate_embeddings([[1.0, 2.0]], 1, 2)
    with pytest.raises(EmbeddingError):
        validate_embeddings([[1.0]], 2, 1)
    with pytest.raises(EmbeddingError):
        validate_embeddings([[1.0]], 1, 2)
    with pytest.raises(EmbeddingError):
        validate_embeddings([[float("nan"), 1.0]], 1, 2)
    with pytest.raises(EmbeddingError):
        validate_embeddings([[]], 1, None)


def test_batching_and_dimension(stub):
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", batch_size=2)
    vectors = emb.embed_documents(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5 and emb.dimension == 4
    assert [len(b) for b in emb._model.encoded] == [2, 2, 1]


def test_empty_input(stub):
    emb = SentenceTransformerEmbedder("m")
    assert emb.embed_documents([]) == []
    with pytest.raises(EmbeddingError):
        emb.embed_documents(["ok", "  "])
    with pytest.raises(EmbeddingError):
        emb.embed_query("")


def test_query_prefix_for_bge_only(stub):
    bge = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    bge.embed_query("risks?")
    assert bge._model.encoded[-1] == [BGE_QUERY_PREFIX + "risks?"]
    other = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    other.embed_query("risks?")
    assert other._model.encoded[-1] == ["risks?"]
    custom = SentenceTransformerEmbedder("x", query_prefix="query: ")
    custom.embed_query("q")
    assert custom._model.encoded[-1] == ["query: q"]


def test_retry_then_success_and_final_failure(stub):
    emb = SentenceTransformerEmbedder("m")
    emb._model.fail_times = 2
    assert len(emb.embed_documents(["a"])) == 1
    emb._model.fail_times = 10
    with pytest.raises(EmbeddingError, match="after 3 attempts"):
        emb.embed_documents(["a"])


def test_model_load_failure_is_wrapped(monkeypatch):
    module = types.ModuleType("sentence_transformers")

    def boom(*a, **k):
        raise OSError("no network")

    module.SentenceTransformer = boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(EmbeddingError, match="Could not load embedding model"):
        SentenceTransformerEmbedder("m")
