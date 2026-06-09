# core/user_db.py
import asyncio
import os
from datetime import date, datetime
from typing import Any
import logging
import json as _json

import asyncpg
from cachetools import TTLCache
from asyncpg.exceptions import UndefinedColumnError, UndefinedTableError
from dotenv import load_dotenv

load_dotenv()

# 5 dk TTL, 1000 kullanıcı kapasiteli profil önbelleği
_profile_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

_profiles: dict[str, dict[str, Any]] = {}

# DB yoksa bellek-içi stub devreye girer
_analyses: dict[str, list[dict[str, Any]]] = {}

_doc_analyses: dict[str, dict[str, Any]] = {}
_user_doc_analyses: dict[str, list[str]] = {}

logger = logging.getLogger(__name__)


def _use_db() -> bool:
    """DB baglantisi aktif mi?"""
    return bool(os.getenv("DATABASE_URL"))


def _has_profile_payload(profile: dict[str, Any]) -> bool:
    """Profilin AI tarafinda anlamli veri tasiyip tasimadigini kontrol et."""
    if not profile:
        return False
    return any(
        [
            bool(profile.get("kronik")),
            bool(profile.get("alerji")),
            bool(profile.get("ilaclar")),
            bool(profile.get("yas")),
            bool(profile.get("cinsiyet")),
            bool(profile.get("boy_cm")),
            bool(profile.get("kilo_kg")),
            bool(profile.get("kan_grubu")),
            profile.get("sigara") is not None,
        ]
    )


def _safe_int(user_id: str) -> int | None:
    """Güvenli int dönüşümü; başarısız olursa None döndürür."""
    try:
        return int(user_id)
    except (TypeError, ValueError):
        return None


# ─── Profil ─────────────────────────────────────────────────────────────────

def set_user_profile(user_id: str, profile: dict[str, Any]) -> None:
    """Stub: test için manuel profil ekle."""
    _profiles[user_id] = profile
    # Cache'i invalide et
    _profile_cache.pop(user_id, None)


async def get_user_profile(user_id: str) -> dict[str, Any]:
    # Önbellek kontrolü
    if user_id in _profile_cache:
        logger.info("user_profile cache_hit user_id=%s", user_id)
        return _profile_cache[user_id]

    if not _use_db():
        result = _profiles.get(user_id, {})
        if _has_profile_payload(result):
            _profile_cache[user_id] = result
        return result

    uid = _safe_int(user_id)
    if uid is None:
        return {}

    from core.database import get_pool
    pool = await get_pool()

    async def _fetch_base() -> asyncpg.Record | None:
        try:
            return await pool.fetchrow(
                """
                SELECT EXTRACT(YEAR FROM AGE(birth_date))::INT AS yas, gender AS cinsiyet,
                       height     AS boy_cm,
                       weight     AS kilo_kg,
                       blood_type AS kan_grubu,
                       smoker
                FROM db_users
                WHERE id = $1
                """,
                uid,
            )
        except UndefinedTableError:
            logger.warning("user_profile base table missing (db_users) user_id=%s", user_id)
            return None

    async def _fetch_kronik() -> list:
        try:
            return await pool.fetch(
                """
                SELECT cd.label_tr AS name
                FROM user_chronic_diseases ucd
                         JOIN chronic_disease cd ON cd.id = ucd.chronic_disease_id
                WHERE ucd.user_id = $1
                """,
                uid,
            )
        except UndefinedTableError:
            logger.warning("user_profile chronic tables missing user_id=%s", user_id)
            return []

    async def _fetch_alerji() -> list:
        try:
            return await pool.fetch(
                """
                SELECT a.name_tr AS name
                FROM user_allergens ua
                         JOIN allergens a ON a.id = ua.allergen_id
                WHERE ua.user_id = $1
                """,
                uid,
            )
        except UndefinedTableError:
            logger.warning("user_profile allergen tables missing user_id=%s", user_id)
            return []

    async def _fetch_ilac() -> list:
        try:
            return await pool.fetch(
                """
                SELECT COALESCE(NULLIF(TRIM(um.note), ''), 'İlaç') AS ilac_adi,
                       um.dosage                                  AS doz,
                       um.reason                                  AS kullanim_nedeni
                FROM user_medication um
                WHERE um.user_id = $1
                  AND COALESCE(um.active, TRUE) = TRUE
                  AND (um.start_date IS NULL OR um.start_date <= CURRENT_DATE)
                  AND (um.end_date IS NULL OR um.end_date >= CURRENT_DATE)
                ORDER BY um.created_at DESC, um.id DESC
                """,
                uid,
            )
        except UndefinedTableError:
            logger.warning("user_profile medication tables missing user_id=%s", user_id)
            return []
        except UndefinedColumnError as exc:
            logger.warning("user_profile medication columns differ; skipping medications user_id=%s error=%s", user_id, exc)
            return []

    row, kronik_rows, alerji_rows, ilac_rows = await asyncio.gather(
        _fetch_base(), _fetch_kronik(), _fetch_alerji(), _fetch_ilac()
    )

    if row is None:
        return {}
    if not row:
        logger.info("user_profile user_not_found user_id=%s", user_id)
        return {}

    ilaclar: list[str] = []
    for r in ilac_rows:
        ilac_adi = (r["ilac_adi"] or "").strip()
        if not ilac_adi:
            continue
        doz = (r["doz"] or "").strip()
        kullanim_nedeni = (r["kullanim_nedeni"] or "").strip()

        ilac_metin = f"{ilac_adi} ({doz})" if doz else ilac_adi
        if kullanim_nedeni:
            ilac_metin += f" - neden: {kullanim_nedeni}"
        ilaclar.append(ilac_metin)

    result = {
        "yas": row["yas"],
        "cinsiyet": row["cinsiyet"],
        "boy_cm": row["boy_cm"],
        "kilo_kg": row["kilo_kg"],
        "kan_grubu": row["kan_grubu"],
        "sigara": row["smoker"],
        "kronik": [r["name"] for r in kronik_rows],
        "alerji": [r["name"] for r in alerji_rows],
        "ilaclar": ilaclar,
    }
    logger.info(
        "user_profile loaded user_id=%s kronik=%d alerji=%d ilac=%d",
        user_id, len(result["kronik"]), len(result["alerji"]), len(result["ilaclar"]),
    )
    if _has_profile_payload(result):
        _profile_cache[user_id] = result
    return result


async def get_recent_blood_sugar_measurements(
    user_id: str,
    days: int = 7,
    limit: int = 256,
) -> list[dict[str, Any]]:
    """Kullanicinin son N gundeki kan sekeri olcumlerini dondurur."""
    if not _use_db():
        return []

    uid = _safe_int(user_id)
    if uid is None:
        return []

    safe_days = max(1, min(days, 30))
    safe_limit = max(1, min(limit, 1000))

    from core.database import get_pool
    pool = await get_pool()

    try:
        rows = await pool.fetch(
            """
            SELECT value, unit, measured_at, context, level
            FROM blood_sugar_measurements
            WHERE user_id = $1
              AND measured_at >= (NOW() - ($2 * INTERVAL '1 day'))
            ORDER BY measured_at ASC, id ASC
            LIMIT $3
            """,
            uid,
            safe_days,
            safe_limit,
        )
    except UndefinedTableError:
        logger.warning("blood_sugar_measurements table not found; skipping fetch")
        return []

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        measured_at = item.get("measured_at")
        if isinstance(measured_at, datetime):
            item["measured_at"] = measured_at.isoformat()
        result.append(item)
    return result


# ─── Analiz Geçmişi ─────────────────────────────────────────────────────────

async def get_recent_analyses(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    if not _use_db():
        return _analyses.get(user_id, [])[-limit:]

    uid = _safe_int(user_id)
    if uid is None:
        return []

    from core.database import get_pool
    pool = await get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT tarih, semptom_ozeti, olasi_taniler
            FROM zenvora_analyses
            WHERE user_id = $1
            ORDER BY created_at DESC LIMIT $2
            """,
            uid, limit,
        )
    except UndefinedTableError:
        return []
    result: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        tarih = item.get("tarih")
        if isinstance(tarih, (date, datetime)):
            item["tarih"] = tarih.strftime("%d.%m.%Y")
        result.append(item)
    return result


async def save_analysis(
        user_id: str,
        semptomlar: list[str],
        olasi_taniler: list[str],
        aksiyon: str | None,
) -> None:
    """Tek bir tıbbi yanıtı yapılandırılmış olarak kaydet (sadece tibbi=True olanlar)."""
    aksiyon_suffix = f" (öneri: {aksiyon})" if aksiyon else ""
    today = datetime.now().date()
    record = {
        "tarih": datetime.now().strftime("%d.%m.%Y"),
        "semptom_ozeti": ", ".join(semptomlar)[:200],
        "olasi_taniler": (", ".join(olasi_taniler) + aksiyon_suffix)[:200],
    }

    if not _use_db():
        _analyses.setdefault(user_id, []).append(record)
        return

    uid = _safe_int(user_id)
    if uid is None:
        return

    from core.database import get_pool
    pool = await get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO zenvora_analyses (user_id, tarih, semptom_ozeti, olasi_taniler, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            """,
            uid, today, record["semptom_ozeti"], record["olasi_taniler"],
        )
    except UndefinedTableError:
        return

# ─── Belge Analizleri ────────────────────────────────────────────────────────

async def save_document_analysis(
    user_id: str,
    analysis_id: str,
    document_type: str,
    severity: str,
    file_name: str | None,
    summary: str,
    findings: list[dict],
    recommendations: list[str],
    type_specific: dict,
) -> None:
    """Belge analizini document_analyses tablosuna kaydet."""
    record: dict[str, Any] = {
        "analysis_id": analysis_id,
        "document_type": document_type,
        "severity": severity,
        "file_name": file_name,
        "summary": summary,
        "findings": findings,
        "recommendations": recommendations,
        "type_specific": type_specific,
        "created_at": datetime.now().isoformat(),
    }

    if not _use_db():
        _doc_analyses[analysis_id] = {**record, "user_id": user_id}
        _user_doc_analyses.setdefault(user_id, []).append(analysis_id)
        return

    uid = _safe_int(user_id)
    if uid is None:
        return

    from core.database import get_pool
    pool = await get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO document_analyses
    (analysis_id, user_id, document_type, severity, file_name, summary,
     findings, recommendations, type_specific, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, NOW())
""",
analysis_id, uid, document_type, severity, file_name, summary,
_json.dumps(findings, ensure_ascii=False),
_json.dumps(recommendations, ensure_ascii=False),
_json.dumps(type_specific, ensure_ascii=False),
        )
    except UndefinedTableError:
        logger.warning("document_analyses table not found; skipping save")


async def get_document_analysis(analysis_id: str, user_id: str) -> dict[str, Any] | None:
    """Tek bir analizi döndür; user_id eşleşmezse None."""
    if not _use_db():
        record = _doc_analyses.get(analysis_id)
        if record and str(record.get("user_id")) == str(user_id):
            return record
        return None

    uid = _safe_int(user_id)
    if uid is None:
        return None

    from core.database import get_pool
    pool = await get_pool()
    try:
        row = await pool.fetchrow(
            """
            SELECT analysis_id::text, document_type, severity, summary,
                   findings, recommendations, type_specific, created_at
            FROM document_analyses
            WHERE analysis_id = $1 AND user_id = $2
            """,
            analysis_id, uid,
        )
    except UndefinedTableError:
        return None

    if not row:
        return None

    item = dict(row)
    for key in ("findings", "recommendations", "type_specific"):
        raw = item.get(key)
        if isinstance(raw, str):
            item[key] = _json.loads(raw)
    created = item.get("created_at")
    if isinstance(created, datetime):
        item["created_at"] = created.isoformat()
    return item


async def get_user_document_analyses(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Kullanıcının tüm belge analizlerini (özet liste) döndür."""
    if not _use_db():
        ids = _user_doc_analyses.get(user_id, [])[-limit:]
        return [
            {
                "analysis_id": _id,
                "document_type": _doc_analyses[_id]["document_type"],
                "created_at": _doc_analyses[_id]["created_at"],
                "summary": _doc_analyses[_id]["summary"],
            }
            for _id in reversed(ids)
            if _id in _doc_analyses
        ]

    uid = _safe_int(user_id)
    if uid is None:
        return []

    from core.database import get_pool
    pool = await get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT analysis_id::text, document_type, created_at, summary
            FROM document_analyses
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            uid, limit,
        )
    except UndefinedTableError:
        return []

    result: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        created = item.get("created_at")
        if isinstance(created, datetime):
            item["created_at"] = created.isoformat()
        result.append(item)
    return result



async def save_analysis_history(
        user_id: str,
        session_id: str,
        diagnosis: dict,
) -> None:
    """Oturum kapanışında üretilen tanı JSON'ını analysis_histories tablosuna yazar."""
    if not diagnosis:
        return

    payload = dict(diagnosis)
    payload.setdefault("session_id", session_id)

    if not _use_db():
        return

    uid = _safe_int(user_id)
    if uid is None:
        return

    from core.database import get_pool
    pool = await get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO analysis_histories (user_id, diagnosis_json, created_at)
            VALUES ($1, $2::jsonb, NOW())
            """,
            uid, _json.dumps(payload, ensure_ascii=False),
        )
    except UndefinedTableError:
        logger.warning("analysis_histories table not found; skipping history save")
        return


# Aksiyon öncelik sıralaması — oturum özetinde en yüksek seviyeyi bul
_AKSIYON_PRIORITY: dict[str, int] = {
    "acil": 4, "doktor": 3, "recetesiz": 2, "ev": 1
}



async def save_session_summary(user_id: str, records: list[dict]) -> None:
    """Oturum kapanırken tüm tıbbi kayıtları tek bir özet kayda indirir.

    Geriye uyumluluk için DB şeması korunur: semptom/olası tanı listesi
    birleştirilip tek satır olarak yazılır. Mobil için üretilen zengin
    JSON çıktısı bu kayıttan bağımsızdır (routes.py tarafından dönülür).
    """
    if not records:
        return

    # Tüm semptom ve tanıları birleştir, tekrarları kaldır
    all_semptomlar: list[str] = []
    all_taniler: list[str] = []
    max_priority = 0
    max_aksiyon: str | None = None

    for r in records:
        all_semptomlar.extend(r.get("semptomlar", []))
        all_taniler.extend(r.get("olasi_taniler", []))
        aksiyon = r.get("aksiyon")
        if aksiyon and _AKSIYON_PRIORITY.get(aksiyon, 0) > max_priority:
            max_priority = _AKSIYON_PRIORITY[aksiyon]
            max_aksiyon = aksiyon

    # Sırayı koruyarak tekilleştir
    unique_semptomlar = list(dict.fromkeys(all_semptomlar))
    unique_taniler = list(dict.fromkeys(all_taniler))

    aksiyon_suffix = f" (öneri: {max_aksiyon})" if max_aksiyon else ""
    today = datetime.now().date()
    summary_record = {
        "tarih": datetime.now().strftime("%d.%m.%Y"),
        "semptom_ozeti": ("[Oturum] " + ", ".join(unique_semptomlar))[:200],
        "olasi_taniler": (", ".join(unique_taniler) + aksiyon_suffix)[:200],
    }

    if not _use_db():
        _analyses.setdefault(user_id, []).append(summary_record)
        return

    uid = _safe_int(user_id)
    if uid is None:
        return

    from core.database import get_pool
    pool = await get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO zenvora_analyses (user_id, tarih, semptom_ozeti, olasi_taniler, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            """,
            uid, today,
            summary_record["semptom_ozeti"],
            summary_record["olasi_taniler"],
        )
    except UndefinedTableError:
        return
