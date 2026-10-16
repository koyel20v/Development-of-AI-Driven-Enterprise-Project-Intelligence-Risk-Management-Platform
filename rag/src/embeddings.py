"""Embedding generation.

No embedding provider existed in the project, so this module introduces exactly
one: a **local** ``sentence-transformers`` model (default ``BAAI/bge-small-en-v1.5``,
384 dimensions, 512-token limit). It needs no API key and no network access after
the first model download, and it keeps confidential project documents on your
machine.

The same :class:`SentenceTransformerEmbedder` instance (hence the same model) is
used for documents and queries. Vectors are L2-normalised so cosine distance in
ChromaDB behaves as expected.

Any object with ``model_name``, ``dimension``, ``embed_documents`` and
``embed_query`` satisfies the :class:`Embedder` protocol, which is how tests
inject a fake embedder.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_MAX_ATTEMPTS = 3


class EmbeddingError(RuntimeError):
    """Embedding generation failed or produced invalid output."""


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


def validate_embeddings(vectors: Sequence[Sequence[float]], expected_count: int, expected_dim: Optional[int]) -> None:
    """Check count, dimension and finiteness of a batch of vectors.

    Raises:
        EmbeddingError: on any inconsistency.
    """
    if len(vectors) != expected_count:
        raise EmbeddingError(f"Expected {expected_count} embeddings, received {len(vectors)}")
    for index, vector in enumerate(vectors):
        if expected_dim is not None and len(vector) != expected_dim:
            raise EmbeddingError(
                f"Embedding {index} has dimension {len(vector)}, expected {expected_dim}"
            )
        if len(vector) == 0:
            raise EmbeddingError(f"Embedding {index} is empty")
        if not all(math.isfinite(float(x)) for x in vector):
            raise EmbeddingError(f"Embedding {index} contains NaN or infinite values")


class SentenceTransformerEmbedder:
    """Local embedding model wrapper with batching, retries and output validation."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        if query_prefix is None:
            lowered = model_name.lower()
            query_prefix = BGE_QUERY_PREFIX if "bge" in lowered and "-en" in lowered else ""
        self.query_prefix = query_prefix

        try:
            logger.info("Loading embedding model %s", model_name)
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model {model_name!r}: {exc}. "
                "The first run needs internet access to download the model."
            ) from exc

        dimension = self._model.get_sentence_embedding_dimension()
        if not dimension:
            dimension = len(self._encode(["dimension probe"])[0])
        self.dimension = int(dimension)
        self.max_seq_length = getattr(self._model, "max_seq_length", None)

    # ------------------------------------------------------------------ internals
    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                array = self._model.encode(
                    list(texts),
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return [[float(x) for x in row] for row in array]
            except Exception as exc:  # e.g. out-of-memory on a large batch
                last_error = exc
                logger.warning("Embedding attempt %d/%d failed: %s", attempt, _MAX_ATTEMPTS, exc)
                time.sleep(min(2 ** (attempt - 1), 4))
        raise EmbeddingError(f"Embedding failed after {_MAX_ATTEMPTS} attempts: {last_error}") from last_error

    @staticmethod
    def _check_texts(texts: Sequence[str]) -> None:
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError(f"Cannot embed empty or non-string text at index {index}")

    # ------------------------------------------------------------------ public API
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed document chunks in batches. ``[]`` in, ``[]`` out."""
        if not texts:
            return []
        self._check_texts(texts)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._encode(texts[start : start + self.batch_size]))
        validate_embeddings(vectors, len(texts), self.dimension)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a user question with the same model (plus the model's query prefix, if any)."""
        self._check_texts([text])
        vector = self._encode([self.query_prefix + text.strip()])
        validate_embeddings(vector, 1, self.dimension)
        return vector[0]


def create_embedder(settings) -> SentenceTransformerEmbedder:
    """Build the configured embedder from :class:`config.RAGSettings`."""
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        query_prefix=settings.embedding_query_prefix,
    )
