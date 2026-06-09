"""ChromaDB + E5 tabanlı RAG deposu — dependency injection için class API.

Kullanım (production):
    from core.rag_store import get_rag_store
    store = get_rag_store()
    results = store.search("baş ağrısı", top_k=3)
    for r in results:
        print(r.text, r.score)

Kullanım (FastAPI):
    from fastapi import Depends
    from core.rag_store import RAGStore, get_rag_store

    @router.get("/search")
    def search(q: str, store: RAGStore = Depends(get_rag_store)):
        return store.search(q)

Test (dependency override):
    app.dependency_overrides[get_rag_store] = lambda: FakeStore()
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.rag_embedder import E5Embedder


@dataclass(frozen=True)
class SearchResult:
    """Tekil arama sonucu — immutable DTO."""
    text: str
    score: float
    metadata: dict


def _detect_category(doc_id: str) -> str:
    """Dosya adından kategori tahmin et — metadata filtrelemede kullanılır."""
    if doc_id.startswith("lab_"):
        return "lab"
    if doc_id.startswith("drug_"):
        return "drug"
    return "disease"


class RAGStore:
    """ChromaDB + E5 üzerinde semantik arama deposu."""

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma_db",
        collection_name: str = "medical_docs",
        embed_model: E5Embedder | None = None,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
    ) -> None:
        self._embed_model = embed_model or E5Embedder()
        # LlamaIndex global ayarları — LLM yok, sadece retrieval
        Settings.embed_model = self._embed_model
        Settings.llm = None

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"},
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            self._vector_store, storage_context=self._storage,
        )
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 2,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Semantik arama — top_k benzer chunk'ı SearchResult listesi olarak döner."""
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        if category:
            nodes = [n for n in nodes if n.metadata.get("category") == category]
        return [
            SearchResult(
                text=n.text,
                score=float(getattr(n, "score", 0.0) or 0.0),
                metadata=dict(n.metadata),
            )
            for n in nodes
        ]

    def add_document(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> None:
        """Tek bir dökümanı indeksle."""
        meta = dict(metadata or {})
        meta.setdefault("category", _detect_category(doc_id))
        meta["doc_id"] = doc_id
        doc = Document(text=text, metadata=meta, id_=doc_id)
        nodes = self._splitter.get_nodes_from_documents([doc])
        for n in nodes:
            n.metadata.update(meta)
        self._index.insert_nodes(nodes)

    def warmup(self) -> None:
        """Embedder belleğe yüklendiğinden emin ol (constructor'da olur, idempotent)."""
        # E5Embedder() çağrısı zaten _get_st_model'i tetikler; burası no-op
        pass

    def count(self) -> int:
        """Indekslenen toplam node sayısı."""
        return int(self._collection.count())

    def load_folder(self, folder: str | Path) -> int:
        """Klasördeki .txt dosyalarını indeksle; zaten varsa atla.

        Döner: yeni eklenen doküman sayısı.
        """
        path = Path(folder)
        if not path.exists():
            return 0
        existing_ids = set(self._collection.get()["ids"])
        added = 0
        for file in path.glob("*.txt"):
            already_indexed = any(eid.startswith(file.stem) for eid in existing_ids)
            if already_indexed:
                continue
            self.add_document(
                text=file.read_text(encoding="utf-8"),
                doc_id=file.stem,
                metadata={"source": file.name},
            )
            added += 1
        return added

    # ── Backwards-compat: internal state için read accessor'lar ───────────
    # (Geriye uyumlu core/rag.py shim'i ve test fixture'ları kullanır.)

    @property
    def collection(self):
        return self._collection

    @property
    def index(self):
        return self._index


# ─── FastAPI dependency factory ───────────────────────────────────────────

_default_store: RAGStore | None = None


def get_rag_store() -> RAGStore:
    """FastAPI Depends() ile inject edilen process-wide singleton.

    Test sırasında app.dependency_overrides[get_rag_store] = lambda: FakeStore()
    ile override edilebilir.
    """
    global _default_store
    if _default_store is None:
        _default_store = RAGStore()
    return _default_store


def reset_default_store() -> None:
    """Test yardımcısı — singleton'u sıfırla (testler arası izolasyon için)."""
    global _default_store
    _default_store = None
