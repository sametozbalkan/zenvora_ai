"""Test konfigürasyonu — fixtures & izolasyon.

Test stratejisi:
  - core.rag module load sırasında ChromaDB PersistentClient açıyor ve
    Settings.embed_model'i set ediyor. Bu yan etkileri test-by-test
    izole etmek için module-level singleton'lar monkeypatch ile swap edilir.
  - Embedding modeli ağır (≈400 MB), unit testlerde mock; integration
    testlerde session-scope ile bir kez yüklenir.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest

# Test ortamı için DB'yi devre dışı bırak (user_db stub moduna düşsün)
os.environ.pop("DATABASE_URL", None)
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test")

# Proje kökünü sys.path'e ekle (pytest tests/ klasöründen koşulduğunda)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─── Mock fixtures (unit testler için — gerçek DB yok) ────────────────────────


@pytest.fixture
def mock_chroma_collection() -> MagicMock:
    """ChromaDB collection mock'u — get/add/query metodları."""
    col = MagicMock(name="ChromaCollection")
    col.get.return_value = {"ids": []}
    col.count.return_value = 0
    return col


@pytest.fixture
def mock_st_model() -> MagicMock:
    """SentenceTransformer mock'u — encode() numpy array döner (gerçek API ile uyumlu).

    core.rag `.tolist()` çağırdığı için ndarray döndürmek şart.
    """
    import numpy as np

    model = MagicMock(name="SentenceTransformer")

    def _encode(text, normalize_embeddings=True):
        if isinstance(text, list):
            return np.asarray([_single_vec(t) for t in text], dtype=np.float32)
        return np.asarray(_single_vec(text), dtype=np.float32)

    def _single_vec(text: str) -> list[float]:
        # Deterministik 8-d vektör; aynı input → aynı vektör (cache test için)
        h = abs(hash(text))
        raw = [((h >> (i * 4)) & 0xF) / 15.0 for i in range(8)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    model.encode = MagicMock(side_effect=_encode)
    return model


@pytest.fixture
def patched_rag(monkeypatch, mock_chroma_collection, mock_st_model):
    """core.rag modülünün global state'ini mock'larla swap eder.

    Test sonunda monkeypatch otomatik geri yükler.
    """
    import core.rag as rag

    # SentenceTransformer'ı mock'la — encode() gerçek modeli çağırmasın
    monkeypatch.setattr(rag, "_ST_MODEL", mock_st_model)

    # _get_st_model her zaman mock'u dönsün
    monkeypatch.setattr(rag, "_get_st_model", lambda: mock_st_model)

    # _chroma_collection'ı mock'la
    monkeypatch.setattr(rag, "_chroma_collection", mock_chroma_collection)

    # _index retriever'ını mock'la — retrieve() çağrısı kontrol edilebilir
    mock_retriever = MagicMock(name="LlamaIndexRetriever")
    mock_retriever.retrieve.return_value = []

    mock_index = MagicMock(name="VectorStoreIndex")
    mock_index.as_retriever.return_value = mock_retriever

    monkeypatch.setattr(rag, "_index", mock_index)

    yield rag, mock_retriever, mock_st_model


# ─── Integration fixture: gerçek ChromaDB ama temp dizin ──────────────────────


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Iterator[Path]:
    """İzole, test-spesifik ChromaDB persist dizini."""
    d = tmp_path / "chroma_test"
    d.mkdir()
    yield d


@pytest.fixture
def real_rag_with_temp_db(monkeypatch, temp_chroma_dir: Path, mock_st_model):
    """core.rag shim'ini gerçek ChromaDB (temp dir) + mock embedder'lı RAGStore'a bağlar.

    Embedder mock çünkü gerçek E5 modeli 400MB+ ve yavaş.
    ChromaDB ve LlamaIndex davranışı gerçek — sadece embedding hesaplaması mock'lanır.
    """
    from core import rag, rag_embedder, rag_store

    # Embedder mock — Settings.embed_model = E5Embedder() RAGStore.__init__'te yapılır
    monkeypatch.setattr(rag_embedder, "_ST_MODEL", mock_st_model)
    monkeypatch.setattr(rag_embedder, "_get_st_model", lambda: mock_st_model)
    # Default store singleton'unu sıfırla — testler arası izolasyon
    monkeypatch.setattr(rag_store, "_default_store", None)

    # RAGStore üzerinden kur — Settings.embed_model otomatik set edilir
    store = rag_store.RAGStore(
        persist_dir=str(temp_chroma_dir),
        collection_name=f"test_{temp_chroma_dir.name}",
    )

    # Shim'in module-level state'ini bu store'a bağla
    monkeypatch.setattr(rag, "_chroma_collection", store._collection)
    monkeypatch.setattr(rag, "_vector_store", store._vector_store)
    monkeypatch.setattr(rag, "_storage_context", store._storage)
    monkeypatch.setattr(rag, "_index", store._index)

    yield rag
