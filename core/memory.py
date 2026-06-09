import os
from collections import deque, OrderedDict

from cachetools import TTLCache

# Belleğe alınan ham dosya metni için tavan — bellek kullanımını koru.
MAX_RAW_FILE_CHARS = 30_000
# Soru-odaklı özet önbelleğinde tutulan girdi sayısı.
MAX_FILE_SUMMARY_CACHE = 6

# Maksimum sohbet turu (1 tur = user + assistant). Env'den override edilebilir.
DEFAULT_MAX_TURNS = int(os.getenv("ZENVORA_MEMORY_TURNS", "5"))


class ConversationMemory:
    """Session başına konuşma + ham dosya metni + soru-odaklı özet önbelleği +
    oturum boyunca biriken yapılandırılmış analiz kayıtları.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        # Maksimum 5 tur (10 mesaj) — Groq 6K TPM limitine karşı bütçe korur
        self.messages: deque[dict] = deque(maxlen=max_turns * 2)
        # Dosya: ham metin tek sefer set edilir; özetler talebe göre üretilip cache'lenir
        self.file_raw: str = ""
        self.file_name: str = ""
        # query_key → focused summary; LRU davranışı için OrderedDict
        self.file_summaries: "OrderedDict[str, str]" = OrderedDict()
        # Profil sahibi — DELETE endpoint'i için session'a bağlı tutulur
        self.user_id: str = ""
        # Oturumdaki tıbbi analiz kayıtları (yalnızca tibbi=True olanlar)
        self.session_records: list[dict] = []

    # ── Konuşma ──────────────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return list(self.messages)

    # ── Dosya: ham metin ─────────────────────────────────────────────────────

    def set_file_raw(self, raw_text: str, file_name: str = "") -> None:
        """Yüklenen dosyanın ham metnini tut; özet üretme.

        Özet, kullanıcı bir şey sorduğunda soru-odaklı olarak üretilir.
        """
        self.file_raw = (raw_text or "")[:MAX_RAW_FILE_CHARS]
        self.file_name = file_name
        self.file_summaries.clear()

    def has_file(self) -> bool:
        return bool(self.file_raw)

    def get_file_raw(self) -> str:
        return self.file_raw

    # ── Dosya: soru-odaklı özet önbelleği ────────────────────────────────────

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").lower().split())[:80]

    def get_cached_summary(self, query: str) -> str:
        key = self._normalize_query(query)
        if key in self.file_summaries:
            # LRU: en sonda kalsın
            self.file_summaries.move_to_end(key)
            return self.file_summaries[key]
        return ""

    def cache_summary(self, query: str, summary: str) -> None:
        if not summary:
            return
        key = self._normalize_query(query)
        self.file_summaries[key] = summary
        self.file_summaries.move_to_end(key)
        while len(self.file_summaries) > MAX_FILE_SUMMARY_CACHE:
            self.file_summaries.popitem(last=False)

    # ── Analiz kayıtları ─────────────────────────────────────────────────────

    def add_analysis_record(self, record: dict) -> None:
        """Tıbbi içerik içeren bir yanıtın yapılandırılmış kaydını ekle."""
        self.session_records.append(record)

    def get_session_records(self) -> list[dict]:
        return list(self.session_records)

    # ── Temizle ──────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.messages.clear()
        self.file_raw = ""
        self.file_name = ""
        self.file_summaries.clear()
        self.session_records = []
        # user_id kasıtlı tutulur


# Her oturum için ayrı hafıza — 15 dakika TTL, max 200 oturum
memories: TTLCache = TTLCache(maxsize=200, ttl=900)


def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in memories:
        memories[session_id] = ConversationMemory()
    return memories[session_id]
