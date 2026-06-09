"""E5 embedding modeli — LlamaIndex BaseEmbedding implementasyonu.

intfloat/multilingual-e5-base modelini SentenceTransformer üzerinden yükler,
"query: " ve "passage: " prefix konvansiyonunu uygular. Query embeddings
LRU cache ile hızlandırılır (aynı kullanıcı sorgusu tekrar gelirse).

Bu modül core/rag_store.py tarafından kullanılır; doğrudan import etmeniz
genelde gerekmez — RAGStore varsayılan olarak E5Embedder() yaratır.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer

# Modül seviyesi singleton — Pydantic v2 class-private attribute karışıklığını önler
_ST_MODEL: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    """SentenceTransformer modelini lazy yükle (ilk çağrıda)."""
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer("intfloat/multilingual-e5-base")
    return _ST_MODEL


@lru_cache(maxsize=1024)
def _cached_query_embedding(query: str) -> tuple[float, ...]:
    """Aynı sorguya tekrar gelirse embedding hesaplamadan dön."""
    vec = _get_st_model().encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()
    return tuple(vec)


class E5Embedder(BaseEmbedding):
    """SentenceTransformer'ı LlamaIndex BaseEmbedding arayüzüne bağlar."""

    def _get_query_embedding(self, query: str) -> List[float]:
        return list(_cached_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return _get_st_model().encode(
            f"passage: {text}", normalize_embeddings=True
        ).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return _get_st_model().encode(prefixed, normalize_embeddings=True).tolist()
