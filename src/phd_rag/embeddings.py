"""Embedding backends: local fastembed (default) or the OpenAI embeddings API."""

from __future__ import annotations

from functools import lru_cache

import faiss
import numpy as np

from .settings import Settings, require_openai_key


@lru_cache(maxsize=2)
def _fastembed_model(model_name: str):
    # Imported lazily so the OpenAI backend does not need fastembed installed.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def _normalized(vectors: list) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def embed_passages(settings: Settings, texts: list[str]) -> np.ndarray:
    """Embed document chunks for indexing; returns L2-normalized float32 vectors."""
    if settings.embedding_provider == "fastembed":
        return _normalized(list(_fastembed_model(settings.embedding_model).passage_embed(texts, batch_size=32)))
    from openai import OpenAI

    client = OpenAI(api_key=require_openai_key())
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), 96):
        response = client.embeddings.create(model=settings.embedding_model, input=texts[offset : offset + 96])
        vectors.extend(item.embedding for item in response.data)
    return _normalized(vectors)


def embed_query(settings: Settings, query: str) -> np.ndarray:
    """Embed one search query; returns a (1, dim) L2-normalized float32 matrix."""
    if settings.embedding_provider == "fastembed":
        # BGE models use a query-specific prompt; query_embed applies it.
        return _normalized(list(_fastembed_model(settings.embedding_model).query_embed(query)))
    from openai import OpenAI

    client = OpenAI(api_key=require_openai_key())
    response = client.embeddings.create(model=settings.embedding_model, input=query)
    return _normalized([response.data[0].embedding])
