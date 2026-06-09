"""Geriye uyumluluk shim'i — yeni RAGStore class'a delege eder.

YENİ KULLANIM (refactor sonrası, önerilen):
    from core.rag_store import RAGStore, get_rag_store
    store = get_rag_store()
    results = store.search("...", top_k=3)   # → list[SearchResult]

ESKİ KULLANIM (hâlâ destekleniyor):
    from core.rag import query, add_document, warmup, load_documents_from_folder
    text = query("...", n_results=3)           # → str

Test fixture'ları bu modüldeki `_index`, `_chroma_collection`, `_ST_MODEL`
attribute'lerini monkeypatch ile değiştirebilir; query()/add_document()
fonksiyonları monkeypatch edilmiş state'i öncelikli kullanır.
"""
from __future__ import annotations

from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from core.rag_embedder import (
    E5Embedder,
    _cached_query_embedding,
    _get_st_model,
)
from core.rag_embedder import _ST_MODEL  # re-export — test monkeypatch hedefi
from core.rag_store import (
    RAGStore,
    SearchResult,
    _detect_category,
    get_rag_store,
)

# ── Module-level state (test izolasyonu için expose edilir) ──────────────────
# Production'da lazy yüklenir; test'te monkeypatch ile swap edilir.
_chroma_collection = None
_vector_store = None
_storage_context = None
_index = None
_splitter: SentenceSplitter | None = None


def _get_splitter() -> SentenceSplitter:
    """SentenceSplitter'ı ilk kullanımda kur — modül import maliyetini düşürür."""
    global _splitter
    if _splitter is None:
        _splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    return _splitter


def _ensure_store_initialized() -> RAGStore:
    """Default store'u yükle ve module-level state'leri eşitle (lazy)."""
    global _chroma_collection, _vector_store, _storage_context, _index
    store = get_rag_store()
    if _chroma_collection is None:
        _chroma_collection = store._collection
        _vector_store = store._vector_store
        _storage_context = store._storage
        _index = store._index
    return store


# ── Public API (geriye uyumlu) ───────────────────────────────────────────────


def query(text: str, n_results: int = 2, category: str | None = None) -> str:
    """Eski API: search sonucunu birleştirilmiş string olarak döner.

    [source]\\nchunk_text formatında, chunk'lar arası boş satır.
    """
    # Test ortamı: monkeypatch ile _index swap edilmiş olabilir
    idx = _index
    if idx is None:
        _ensure_store_initialized()
        idx = _index

    retriever = idx.as_retriever(similarity_top_k=n_results)
    nodes = retriever.retrieve(text)

    if category:
        nodes = [n for n in nodes if n.metadata.get("category") == category]

    if not nodes:
        return ""

    return "\n\n".join(
        f"[{n.metadata.get('source', n.metadata.get('doc_id', ''))}]\n{n.text}"
        for n in nodes
    )


def add_document(
    text: str,
    doc_id: str,
    metadata: dict | None = None,
) -> None:
    """Eski API: doküman ekle."""
    meta = dict(metadata or {})
    meta.setdefault("category", _detect_category(doc_id))
    meta["doc_id"] = doc_id

    doc = Document(text=text, metadata=meta, id_=doc_id)
    nodes = _get_splitter().get_nodes_from_documents([doc])
    for n in nodes:
        n.metadata.update(meta)

    # Test ortamı: monkeypatch _index'i mock'lamış olabilir
    idx = _index
    if idx is None:
        _ensure_store_initialized()
        idx = _index
    idx.insert_nodes(nodes)


def warmup() -> None:
    """Embedder'ı belleğe yükle — ilk istek gecikmesini önle."""
    _get_st_model()


def load_documents_from_folder(folder: str = "data/medical_docs") -> None:
    """Klasördeki .txt dosyalarını indeksle; zaten varsa atla."""
    path = Path(folder)
    if not path.exists():
        print(f"⚠ Klasör bulunamadı: {folder}")
        return

    col = _chroma_collection
    if col is None:
        _ensure_store_initialized()
        col = _chroma_collection

    existing_ids = set(col.get()["ids"])

    for file in path.glob("*.txt"):
        already_indexed = any(eid.startswith(file.stem) for eid in existing_ids)
        if already_indexed:
            print(f"⏭ Zaten var: {file.name}")
            continue
        text = file.read_text(encoding="utf-8")
        add_document(text=text, doc_id=file.stem, metadata={"source": file.name})
        print(f"✓ Yüklendi: {file.name}")


# Re-export edilen semboller — eski importlar bozulmasın
__all__ = [
    "E5Embedder",
    "RAGStore",
    "SearchResult",
    "add_document",
    "get_rag_store",
    "load_documents_from_folder",
    "query",
    "warmup",
]
