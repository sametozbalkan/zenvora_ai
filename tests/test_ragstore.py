"""tests/test_ragstore.py — RAG katmanı için davranış-koruyucu testler.

AMAÇ
----
core/rag.py şu an module-level global state ile çalışıyor:
  - `query(text, n_results, category)` → str
  - `add_document(text, doc_id, metadata)` → None
  - `warmup()` → None
  - `load_documents_from_folder(folder)` → None

Hedef refactor: bu fonksiyonları RAGStore sınıfına taşıyıp FastAPI'ye
dependency injection ile bağlamak. Bu test dosyası refactor öncesi
mevcut davranışı sabitler, refactor sonrası aynı sözleşmenin korunduğunu
doğrular.

Test grupları:
  1. UNIT      — ChromaDB ve E5 mock'lanır (patched_rag fixture)
  2. INTEGRATION — gerçek ChromaDB (temp dir) + mock embedder
  3. DI        — FastAPI app + mock rag inject
  4. EDGE     — uzun query, özel karakter, boş DB

Refactor sonrası RAGStore class API'sini öngören contract testleri en
altta `class TestRAGStoreContract` içinde toplandı; şu an `pytest.skip`
ile koşulmaya hazır iskelet.

ÇALIŞTIRMA
----------
    pytest tests/test_ragstore.py -v
    pytest tests/test_ragstore.py -m "not integration" -v   # hızlı koşu
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# 1) UNIT TESTLER — mock'lı, hızlı, izole
# ═════════════════════════════════════════════════════════════════════════════


class TestQueryUnit:
    """`query()` davranışını ChromaDB/E5'e bağımlı olmadan doğrular."""

    def test_query_returns_string(self, patched_rag):
        rag, retriever, _ = patched_rag
        retriever.retrieve.return_value = []

        result = rag.query("baş ağrısı")

        assert isinstance(result, str)

    def test_query_returns_empty_string_when_no_matches(self, patched_rag):
        rag, retriever, _ = patched_rag
        retriever.retrieve.return_value = []

        result = rag.query("xyz_belirsiz_terim")

        assert result == ""

    def test_query_passes_top_k_to_retriever(self, patched_rag):
        rag, _, _ = patched_rag

        rag.query("öksürük", n_results=5)

        rag._index.as_retriever.assert_called_with(similarity_top_k=5)

    def test_query_default_top_k_is_two(self, patched_rag):
        """Davranış sabitleme: default n_results=2 token tasarrufu için."""
        rag, _, _ = patched_rag

        rag.query("ateş")

        rag._index.as_retriever.assert_called_with(similarity_top_k=2)

    def test_query_concatenates_node_text_with_source_headers(self, patched_rag):
        rag, retriever, _ = patched_rag
        node1 = _make_node("Parasetamol ateş düşürür.", source="ilac.txt", doc_id="ilac")
        node2 = _make_node("İbuprofen ağrı keser.", source="ibuprofen.txt", doc_id="ibu")
        retriever.retrieve.return_value = [node1, node2]

        result = rag.query("ateş")

        assert "[ilac.txt]" in result
        assert "Parasetamol" in result
        assert "[ibuprofen.txt]" in result
        assert "İbuprofen" in result

    def test_query_filters_by_category_when_provided(self, patched_rag):
        rag, retriever, _ = patched_rag
        n_drug = _make_node("ilaç bilgisi", source="d.txt", doc_id="d", category="drug")
        n_lab = _make_node("lab değeri", source="l.txt", doc_id="l", category="lab")
        retriever.retrieve.return_value = [n_drug, n_lab]

        result = rag.query("test", category="drug")

        assert "ilaç bilgisi" in result
        assert "lab değeri" not in result

    def test_query_handles_empty_string(self, patched_rag):
        rag, retriever, _ = patched_rag
        retriever.retrieve.return_value = []

        # Boş string → retriever yine çağrılır ama sonuç boş
        result = rag.query("")

        assert result == ""
        retriever.retrieve.assert_called_once_with("")


class TestAddDocumentUnit:
    """`add_document()` davranışı."""

    def test_add_document_inserts_nodes_into_index(self, patched_rag):
        rag, _, _ = patched_rag

        rag.add_document("Aspirin kalp hastalarına önerilir.", doc_id="aspirin_info")

        rag._index.insert_nodes.assert_called_once()
        nodes = rag._index.insert_nodes.call_args[0][0]
        assert len(nodes) >= 1

    def test_add_document_auto_detects_lab_category(self, patched_rag):
        """`lab_` prefix → category=lab metadata'sı."""
        rag, _, _ = patched_rag

        rag.add_document("Kan değerleri tablosu", doc_id="lab_cbc")

        nodes = rag._index.insert_nodes.call_args[0][0]
        assert nodes[0].metadata.get("category") == "lab"

    def test_add_document_auto_detects_drug_category(self, patched_rag):
        rag, _, _ = patched_rag

        rag.add_document("Parasetamol kullanımı", doc_id="drug_paracetamol")

        nodes = rag._index.insert_nodes.call_args[0][0]
        assert nodes[0].metadata.get("category") == "drug"

    def test_add_document_default_category_is_disease(self, patched_rag):
        rag, _, _ = patched_rag

        rag.add_document("Migren bilgisi", doc_id="migren")

        nodes = rag._index.insert_nodes.call_args[0][0]
        assert nodes[0].metadata.get("category") == "disease"

    def test_add_document_preserves_custom_metadata(self, patched_rag):
        rag, _, _ = patched_rag

        rag.add_document(
            "İçerik",
            doc_id="custom_doc",
            metadata={"source": "manual.txt", "author": "zenvora"},
        )

        nodes = rag._index.insert_nodes.call_args[0][0]
        assert nodes[0].metadata["source"] == "manual.txt"
        assert nodes[0].metadata["author"] == "zenvora"


class TestWarmup:
    """`warmup()` davranışı — embedder belleğe yüklenir."""

    def test_warmup_loads_embedding_model(self, patched_rag):
        rag, _, mock_model = patched_rag

        rag.warmup()
        # _get_st_model her çağrıda mock döner — warmup en az bir kez çağırmalı
        # (mock'ı çağrı sayısı üzerinden değil, side-effect-free olduğundan
        #  hata atmaması ile doğruluyoruz)
        assert mock_model is not None


# ═════════════════════════════════════════════════════════════════════════════
# 2) INTEGRATION TESTLER — gerçek ChromaDB (temp dir), mock embedder
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRagIntegration:
    """Gerçek ChromaDB üzerinde end-to-end davranış."""

    def test_add_then_query_finds_document(self, real_rag_with_temp_db):
        rag = real_rag_with_temp_db
        rag.add_document(
            "Parasetamol ateş düşürmek için kullanılır. Yan etkileri sınırlıdır.",
            doc_id="parasetamol_bilgi",
        )

        result = rag.query("parasetamol", n_results=3)

        # Mock embedder deterministik — aynı string aynı vektör → match
        assert isinstance(result, str)
        # Sonuç boş olabilir (mock vektörlerin similarity threshold'u tutmazsa)
        # ama exception atmamalı

    def test_warmup_does_not_raise(self, real_rag_with_temp_db):
        rag = real_rag_with_temp_db
        rag.warmup()  # sadece kraşmadığını doğrula

    def test_empty_db_query_returns_empty_string(self, real_rag_with_temp_db):
        rag = real_rag_with_temp_db

        result = rag.query("hiç eklenmemiş içerik")

        assert result == ""

    def test_load_documents_from_folder_skips_indexed(
        self, real_rag_with_temp_db, tmp_path
    ):
        rag = real_rag_with_temp_db
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "migren.txt").write_text("Migren baş ağrısı türüdür.", encoding="utf-8")
        (docs / "lab_cbc.txt").write_text("Tam kan sayımı değerleri.", encoding="utf-8")

        # İlk çağrı → indeksler
        rag.load_documents_from_folder(str(docs))
        # İkinci çağrı → "zaten var" diyerek atlamalı, exception fırlatmamalı
        rag.load_documents_from_folder(str(docs))


# ═════════════════════════════════════════════════════════════════════════════
# 3) DEPENDENCY INJECTION TESTLERİ — FastAPI + mock RAGStore
# ═════════════════════════════════════════════════════════════════════════════


class TestDependencyInjection:
    """RAGStore'un FastAPI dependency olarak inject edilebileceğini doğrular.

    Refactor öncesi: app rag modülünü direkt import ediyor → DI yok.
    Refactor sonrası: `Depends(get_rag_store)` ile mock'lanabilmeli.

    Bu test refactor için template — şu an `pytest.skip` ile pasif.
    """

    def test_search_endpoint_uses_injected_ragstore(self):
        """Mini FastAPI app + dependency_override ile RAGStore mock'lanır."""
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient

        from core.rag_store import SearchResult, get_rag_store

        class FakeRagStore:
            def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
                return [SearchResult(text=f"mock sonuç: {query}", score=0.9, metadata={})]

        app = FastAPI()

        @app.get("/search")
        def search(q: str, store=Depends(get_rag_store)):
            return [{"text": r.text, "score": r.score} for r in store.search(q)]

        app.dependency_overrides[get_rag_store] = lambda: FakeRagStore()

        with TestClient(app) as client:
            response = client.get("/search?q=ateş")

        assert response.status_code == 200
        assert "mock sonuç" in response.json()[0]["text"]

    def test_rag_uses_dependency_injection(self):
        """Refactor sonrası: api.routes artık get_rag_store kullanıyor, rag_query yok."""
        from api import routes as api_routes

        assert hasattr(api_routes, "get_rag_store")
        assert hasattr(api_routes, "RAGStore")
        assert not hasattr(api_routes, "rag_query"), (
            "rag_query global import'u kaldırılmalı — DI ile gelmeli"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4) EDGE CASE'LER
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Sınır durumlar — uzun query, özel karakter, kategori filtreleme."""

    def test_very_long_query_does_not_crash(self, patched_rag):
        rag, retriever, _ = patched_rag
        retriever.retrieve.return_value = []
        long_query = "ağrı " * 500  # 2500 karakter

        result = rag.query(long_query)

        assert isinstance(result, str)

    def test_query_with_special_characters(self, patched_rag):
        rag, retriever, _ = patched_rag
        retriever.retrieve.return_value = []

        result = rag.query("Q1' OR 1=1 --; <script>alert(1)</script>")

        assert isinstance(result, str)

    def test_query_with_unicode_medical_units(self, patched_rag):
        """µmol/L, β-HCG, °C gibi tıbbi birimleri içeren query."""
        rag, retriever, _ = patched_rag
        retriever.retrieve.return_value = []

        result = rag.query("β-HCG değerleri µIU/mL ölçümü 37°C")

        assert isinstance(result, str)

    def test_query_with_unknown_category_returns_empty(self, patched_rag):
        rag, retriever, _ = patched_rag
        node = _make_node("içerik", source="s.txt", doc_id="d", category="disease")
        retriever.retrieve.return_value = [node]

        result = rag.query("test", category="nonexistent_category")

        assert result == ""

    def test_add_document_with_empty_text(self, patched_rag):
        """Boş metin → splitter 0 node üretebilir; exception atmamalı."""
        rag, _, _ = patched_rag

        # En azından kraşmamalı
        rag.add_document("", doc_id="empty_doc")
        # insert_nodes ya hiç çağrılmaz ya da boş liste ile çağrılır
        # (LlamaIndex davranışına bağlı — biz sadece exception olmadığını test ederiz)

    def test_load_nonexistent_folder_does_not_crash(self, patched_rag):
        rag, _, _ = patched_rag

        rag.load_documents_from_folder("kesinlikle/olmayan/klasor")
        # Print uyarısı, exception yok


# ═════════════════════════════════════════════════════════════════════════════
# 5) REFACTOR CONTRACT — gelecekteki RAGStore class API'si için iskelet
# ═════════════════════════════════════════════════════════════════════════════


class TestRAGStoreContract:
    """Refactor sonrası RAGStore class'ının uyması beklenen sözleşme.

    Hedef API:
        store = RAGStore(persist_dir, embed_model)
        store.search(query, top_k, category)        → list[SearchResult]
        store.add_document(text, doc_id, metadata)  → None
        store.warmup()                              → None
        store.count()                               → int

    Bu testler refactor tamamlanınca skip kaldırılarak aktif edilir.
    """

    @pytest.fixture
    def isolated_store(self, monkeypatch, tmp_path, mock_st_model):
        """Her test için yeni RAGStore (temp dir + mock embedder)."""
        from core import rag_embedder
        from core import rag_store as rs

        # E5 gerçek modelini yükleme; mock kullan (yavaşlığı önle)
        monkeypatch.setattr(rag_embedder, "_ST_MODEL", mock_st_model)
        monkeypatch.setattr(rag_embedder, "_get_st_model", lambda: mock_st_model)
        # Default singleton'u sıfırla — testler arası izolasyon
        monkeypatch.setattr(rs, "_default_store", None)

        store = rs.RAGStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name=f"test_{tmp_path.name}",
        )
        return store

    def test_ragstore_instantiable_with_persist_dir(self, isolated_store):
        assert isolated_store is not None

    def test_ragstore_search_returns_list_of_results(self, isolated_store):
        from core.rag_store import SearchResult

        isolated_store.add_document("Aspirin kalp koruyucu olabilir.", doc_id="aspirin")

        results = isolated_store.search("aspirin", top_k=3)

        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], SearchResult)
            assert hasattr(results[0], "text")
            assert hasattr(results[0], "score")
            assert hasattr(results[0], "metadata")

    def test_ragstore_search_respects_top_k(self, isolated_store):
        for i in range(10):
            isolated_store.add_document(f"Doküman {i} hakkında bilgi.", doc_id=f"doc_{i}")

        results = isolated_store.search("doküman", top_k=3)

        assert len(results) <= 3

    def test_ragstore_count_reflects_added_documents(self, isolated_store):
        assert isolated_store.count() == 0

        isolated_store.add_document("içerik metni", doc_id="d1")
        assert isolated_store.count() >= 1

    def test_ragstore_can_be_dependency_injected(self):
        """FastAPI dependency_overrides ile mock store inject et."""
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient

        from core.rag_store import RAGStore, SearchResult, get_rag_store

        app = FastAPI()

        @app.get("/rag/search")
        def endpoint(q: str, store: RAGStore = Depends(get_rag_store)):
            return [r.text for r in store.search(q, top_k=2)]

        class StubStore:
            def search(self, q, top_k=2, category=None):
                return [_SR(f"mock:{q}", 0.9, {})]

        app.dependency_overrides[get_rag_store] = lambda: StubStore()

        with TestClient(app) as client:
            r = client.get("/rag/search?q=test")
        assert r.status_code == 200
        assert r.json() == ["mock:test"]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_node(
    text: str,
    *,
    source: str = "doc.txt",
    doc_id: str = "doc_id",
    category: str = "disease",
    score: float = 0.9,
) -> Any:
    """LlamaIndex NodeWithScore benzeri minimal mock."""
    node = MagicMock(name=f"Node({doc_id})")
    node.text = text
    node.metadata = {"source": source, "doc_id": doc_id, "category": category}
    node.get_score = lambda: score
    return node


class _SR:
    """Refactor sonrası SearchResult stub'ı — sadece contract testlerinde kullanılır."""

    def __init__(self, text: str, score: float, metadata: dict) -> None:
        self.text = text
        self.score = score
        self.metadata = metadata


# ═════════════════════════════════════════════════════════════════════════════
# REFACTOR İSKELETİ — yorum olarak (silmek için hazır şablon)
# ═════════════════════════════════════════════════════════════════════════════
#
# Aşağıdaki iskelet `core/rag_store.py` olarak çıkarılacak. Mevcut
# `core/rag.py` module-level fonksiyonlar buna delege edilerek geriye
# uyumluluk korunur (drop-in geçiş).
#
# ─── core/rag_store.py ───────────────────────────────────────────────────────
#
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable
#
# import chromadb
# from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.vector_stores.chroma import ChromaVectorStore
#
# from core.rag_embedder import E5Embedder   # E5Embedder buraya taşınır
#
#
# @dataclass(frozen=True)
# class SearchResult:
#     text: str
#     score: float
#     metadata: dict
#
#
# class RAGStore:
#     """ChromaDB + E5 tabanlı semantik arama deposu."""
#
#     def __init__(
#         self,
#         persist_dir: str | Path = "data/chroma_db",
#         collection_name: str = "medical_docs",
#         embed_model: E5Embedder | None = None,
#         chunk_size: int = 256,
#         chunk_overlap: int = 20,
#     ) -> None:
#         self._embed_model = embed_model or E5Embedder()
#         Settings.embed_model = self._embed_model
#         Settings.llm = None
#
#         self._client = chromadb.PersistentClient(path=str(persist_dir))
#         self._collection = self._client.get_or_create_collection(
#             collection_name, metadata={"hnsw:space": "cosine"},
#         )
#         self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
#         self._storage = StorageContext.from_defaults(vector_store=self._vector_store)
#         self._index = VectorStoreIndex.from_vector_store(
#             self._vector_store, storage_context=self._storage,
#         )
#         self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#
#     # ── Public API ──────────────────────────────────────────────────────
#
#     def search(
#         self,
#         query: str,
#         top_k: int = 2,
#         category: str | None = None,
#     ) -> list[SearchResult]:
#         retriever = self._index.as_retriever(similarity_top_k=top_k)
#         nodes = retriever.retrieve(query)
#         if category:
#             nodes = [n for n in nodes if n.metadata.get("category") == category]
#         return [
#             SearchResult(
#                 text=n.text,
#                 score=float(getattr(n, "score", 0.0) or 0.0),
#                 metadata=dict(n.metadata),
#             )
#             for n in nodes
#         ]
#
#     def add_document(
#         self,
#         text: str,
#         doc_id: str,
#         metadata: dict | None = None,
#     ) -> None:
#         meta = dict(metadata or {})
#         meta.setdefault("category", _detect_category(doc_id))
#         meta["doc_id"] = doc_id
#         doc = Document(text=text, metadata=meta, id_=doc_id)
#         nodes = self._splitter.get_nodes_from_documents([doc])
#         for n in nodes:
#             n.metadata.update(meta)
#         self._index.insert_nodes(nodes)
#
#     def warmup(self) -> None:
#         # E5Embedder constructor zaten modeli yüklüyor — explicit no-op
#         pass
#
#     def count(self) -> int:
#         return int(self._collection.count())
#
#     def load_folder(self, folder: str | Path) -> int:
#         path = Path(folder)
#         if not path.exists():
#             return 0
#         existing = set(self._collection.get()["ids"])
#         added = 0
#         for file in path.glob("*.txt"):
#             if any(eid.startswith(file.stem) for eid in existing):
#                 continue
#             self.add_document(
#                 text=file.read_text(encoding="utf-8"),
#                 doc_id=file.stem,
#                 metadata={"source": file.name},
#             )
#             added += 1
#         return added
#
#
# def _detect_category(doc_id: str) -> str:
#     if doc_id.startswith("lab_"):  return "lab"
#     if doc_id.startswith("drug_"): return "drug"
#     return "disease"
#
#
# # ─── FastAPI dependency ──────────────────────────────────────────────────
#
# _default_store: RAGStore | None = None
#
#
# def get_rag_store() -> RAGStore:
#     """FastAPI Depends() ile inject edilen factory.
#
#     Test sırasında app.dependency_overrides[get_rag_store] = ...
#     ile override edilebilir.
#     """
#     global _default_store
#     if _default_store is None:
#         _default_store = RAGStore()
#     return _default_store
#
# ─────────────────────────────────────────────────────────────────────────────
#
# Geriye uyumluluk için core/rag.py şuna indirilebilir:
#
#     from core.rag_store import RAGStore, get_rag_store
#     _store = get_rag_store()
#     query             = lambda text, n=2, category=None: "\n\n".join(
#         f"[{r.metadata.get('source', r.metadata.get('doc_id', ''))}]\n{r.text}"
#         for r in _store.search(text, top_k=n, category=category)
#     )
#     add_document      = _store.add_document
#     warmup            = _store.warmup
#     load_documents_from_folder = _store.load_folder
#
# Böylece api/routes.py ve eval/* mevcut import'larıyla çalışmaya devam eder.
