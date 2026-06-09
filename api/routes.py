import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from copy import deepcopy
import logging
from statistics import mean
from uuid import uuid4


@asynccontextmanager
async def _maybe_acquire(sem: "asyncio.Semaphore | None"):
    """Verilen semaphore varsa acquire eder; yoksa no-op."""
    if sem is None:
        yield
        return
    async with sem:
        yield

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from openai import RateLimitError          # groq yerine openai — DeepSeek uyumlu
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.file_parser import parse_file
from core.llm import (
    analyze_blood_pressure,
    analyze_document,
    chat,
    generate_session_diagnosis,
    summarize_file_for_query,
    summarize_blood_sugar,
    summarize_xray_findings,
)
from core.memory import get_memory
from core.rag_store import RAGStore, SearchResult, get_rag_store
from core.xray_mapper import map_bone_to_analysis, map_chest_to_analysis
from core.user_db import (
    get_document_analysis,
    get_recent_blood_sugar_measurements,
    get_recent_analyses,
    get_user_document_analyses,
    get_user_profile,
    save_analysis,
    save_analysis_history,
    save_document_analysis,
    save_session_summary,
)



router = APIRouter()
logger = logging.getLogger(__name__)


class BloodSugarAiSummaryPayload(BaseModel):
    user_id: int = Field(..., gt=0, description="Kullanici kimligi")


class BloodPressureAnalyzePayload(BaseModel):
    """Tek tansiyon ölçümü analiz isteği — mobil tarafından gönderilir.

    systolic/diastolic mmHg cinsinden zorunlu; heart_rate (nabız) opsiyonel.
    Aralıklar fizyolojik olarak makul sınırlar; bunun dışına çıkan değerler
    422 ile reddedilir.
    """
    user_id: int = Field(..., gt=0, description="Kullanıcı kimliği")
    systolic: int = Field(..., ge=50, le=300, description="Sistolik kan basıncı (mmHg)")
    diastolic: int = Field(..., ge=30, le=220, description="Diastolik kan basıncı (mmHg)")
    heart_rate: int | None = Field(
        None,
        ge=20,
        le=260,
        description="Nabız (bpm) — opsiyonel",
        alias="heartRate",
    )

    model_config = {"populate_by_name": True}


def _hash_user(user_id: str) -> str:
    """Log'larda PII sızıntısını önlemek için user_id'yi kısa hash'le göster."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8]


def validated_user_id_form(user_id: str = Form(..., description="Kullanıcı kimliği (tam sayı)")) -> str:
    """Tek noktadan user_id doğrulaması — Form alanı için FastAPI dependency."""
    cleaned = user_id.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="user_id boş olamaz")
    try:
        int(cleaned)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz user_id")
    return cleaned


def validated_user_id_query(user_id: str = Query(..., description="Kullanıcı kimliği")) -> str:
    """Query parametresi için user_id doğrulaması."""
    cleaned = user_id.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="user_id boş olamaz")
    try:
        int(cleaned)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz user_id")
    return cleaned


def _fallback_blood_sugar_summary(measurements: list[dict]) -> str:
    values = [int(item["value"]) for item in measurements if isinstance(item.get("value"), int)]
    if not values:
        return "Son 7 günde geçerli ölçüm bulunamadı."

    low_count = sum(1 for v in values if v < 70)
    high_count = sum(1 for v in values if v > 180)
    urgent_count = sum(1 for v in values if v < 54 or v >= 300)
    latest_value = values[-1]
    avg = round(mean(values), 1)

    risk = "low"
    if urgent_count > 0:
        risk = "urgent"
    elif high_count >= 3 or low_count >= 2:
        risk = "high"
    elif high_count > 0 or low_count > 0:
        risk = "medium"

    summary = f"Son 7 günde {len(values)} ölçüm var. Ortalama {avg} mg/dL, son ölçüm {latest_value} mg/dL."
    if risk == "urgent":
        summary += " Kritik eşik değerleri saptandı."
    elif risk == "high":
        summary += " Dalgalanma ve hedef dışı degerler sık."

    return summary


def _rate_key(request: Request) -> str:
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_key)




# Rate limit'ler env'den okunur — test için ZENVORA_CHAT_RATE_LIMIT=200/minute gibi yükseltilebilir.
# PROD VARSAYILANLARI: chat=8/minute, analyze=4/minute, clear=3/minute
CHAT_RATE_LIMIT    = os.getenv("ZENVORA_CHAT_RATE_LIMIT",    "8/minute")
ANALYZE_RATE_LIMIT = os.getenv("ZENVORA_ANALYZE_RATE_LIMIT", "4/minute")
CLEAR_RATE_LIMIT   = os.getenv("ZENVORA_CLEAR_RATE_LIMIT",   "3/minute")

MAX_FILE_SIZE_MB    = 2
MAX_XRAY_SIZE_MB    = 20   # DICOM dosyaları sık 5–15 MB arası — ayrı limit
MAX_RAW_TEXT_CHARS  = 30_000  # ~7 500 token — tıbbi raporlar için yeterli

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx", ".xlsx"}
VALID_DOC_TYPES    = {"blood_test", "xray", "health_report"}

XRAY_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}
VALID_XRAY_TYPES        = {"chest", "bone"}

# MURA dataset yalnızca üst ekstremiteyi içerir; alt ekstremite / omurga
# eğitim dağılımı dışındadır. Bone modelini sadece bu bölgelerde kabul ediyoruz.
VALID_BONE_REGIONS: set[str] = {
    "el", "bilek", "dirsek", "on_kol", "ust_kol", "omuz", "parmak",
}
_BONE_REGION_TR = {
    "el":     "El",
    "bilek":  "Bilek",
    "dirsek": "Dirsek",
    "on_kol": "Ön kol",
    "ust_kol": "Üst kol",
    "omuz":   "Omuz",
    "parmak": "Parmak",
}

# X-ray dosyaları için magic-byte imzaları — uzantı tutarlılığı kontrolü
_XRAY_MAGIC_SIGS: dict[str, tuple[bytes, ...]] = {
    ".png":  (b"\x89PNG\r\n\x1a\n",),
    ".jpg":  (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}


def _verify_xray_magic(ext: str, contents: bytes) -> bool:
    """X-ray dosyasının uzantısına göre içerik imzasını doğrular.
    DICOM (.dcm) için: 128 byte preamble + 'DICM' marker."""
    if ext == ".dcm":
        return len(contents) > 132 and contents[128:132] == b"DICM"
    sigs = _XRAY_MAGIC_SIGS.get(ext)
    if not sigs:
        return False
    head = contents[:16]
    return any(head.startswith(s) for s in sigs)

_DOC_TYPE_TR = {
    "blood_test":    "Kan Tahlili",
    "xray":          "Röntgen Raporu",
    "health_report": "Sağlık Raporu",
}


async def _finalize_cleared_session(session_id: str, user_id: str, records: list[dict]) -> None:
    if not records or not user_id:
        return

    diagnosis: dict = {}
    diagnosis_tokens = 0
    has_diagnosis_intent = any(r.get("tani_arayisi") for r in records)
    has_symptoms = any(r.get("semptomlar") for r in records)

    if has_diagnosis_intent and has_symptoms:
        profile: dict = {}
        try:
            profile = await get_user_profile(user_id)
        except Exception as exc:
            logger.warning("profile fetch failed during diagnosis: %s", exc)
        try:
            diagnosis, diagnosis_tokens = await generate_session_diagnosis(records, profile)
        except RateLimitError:
            diagnosis, diagnosis_tokens = {}, 0
        except Exception as exc:
            logger.warning("session diagnosis failed: %s", exc)

    
    

    try:
        await save_session_summary(user_id, records)
    except Exception as exc:
        logger.warning("session summary save failed: %s", exc)

    if diagnosis:
        try:
            await save_analysis_history(user_id, session_id, diagnosis)
        except Exception as exc:
            logger.warning("analysis history save failed: %s", exc)


async def _resolve_file_context(memory, user_message: str) -> tuple[str, int]:
    """Dosya bağlamı + bu çağrıda harcanan token sayısı."""
    if not memory.has_file():
        return "", 0

    cached = memory.get_cached_summary(user_message)
    if cached:
        return cached, 0  # cache hit → token harcanmadı

    tokens = 0
    try:
        focused, tokens = await summarize_file_for_query(memory.get_file_raw(), user_message)
    except RateLimitError:
        focused = memory.get_file_raw()[:1000]
    except Exception as exc:
        logger.warning("file summary failed: %s", exc)
        focused = memory.get_file_raw()[:1000]

    memory.cache_summary(user_message, focused)
    return focused, tokens


def _format_analysis_for_context(analysis: dict) -> str:
    """Analiz dict'ini session memory'de kullanılmak üzere okunabilir metne çevirir."""
    doc_label = _DOC_TYPE_TR.get(analysis.get("document_type", ""), "Tıbbi Belge")
    lines = [
        f"Belge Türü: {doc_label}",
        f"Önem Düzeyi: {analysis.get('severity', 'normal')}",
        f"Özet: {analysis.get('summary', '')}",
    ]

    findings = analysis.get("findings") or []
    if findings:
        lines.append("Bulgular:")
        for f in findings:
            unit = f" {f['unit']}" if f.get("unit") else ""
            flag = f" [{f['status']}]" if f.get("status", "normal") != "normal" else ""
            lines.append(f"  - {f.get('label', '')}: {f.get('value', '')}{unit}{flag}")

    recs = analysis.get("recommendations") or []
    if recs:
        lines.append("Öneriler:")
        for r in recs:
            lines.append(f"  - {r}")

    return "\n".join(lines)


# ─── /analyze ────────────────────────────────────────────────────────────────

@router.post("/analyze")
@limiter.limit(ANALYZE_RATE_LIMIT)
async def analyze_endpoint(
    request: Request,
    user_id: str = Depends(validated_user_id_form),
    document_type: str = Form(..., description="blood_test | xray | health_report"),
    file: UploadFile = File(..., description="Analiz edilecek dosya"),
    xray_type: str | None = Form(None, description="X-ray alt tipi: chest | bone (sadece xray için)"),
    bone_region: str | None = Form(
        None,
        description="Kemik röntgeni bölgesi (xray_type=bone için ZORUNLU): "
                    "el | bilek | dirsek | on_kol | ust_kol | omuz | parmak",
    ),
):
    if document_type not in VALID_DOC_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Geçersiz document_type. İzin verilenler: {', '.join(sorted(VALID_DOC_TYPES))}",
        )

    from pathlib import Path as _Path
    ext = _Path(file.filename or "").suffix.lower()


    # ─── XRAY DALI ───────────────────────────────────────────────────────────
    if document_type == "xray":
        if xray_type is None or xray_type not in VALID_XRAY_TYPES:
            raise HTTPException(
                status_code=422,
                detail="xray_type zorunludur: chest veya bone",
            )

        # Bone modelinin MURA dağılımına bağlılığını koru — kapsam dışı bölgelerde
        # model güvenilir çalışmaz; baştan reddet.
        if xray_type == "bone":
            if not bone_region or bone_region not in VALID_BONE_REGIONS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "bone_region zorunludur ve şu değerlerden biri olmalı: "
                        f"{', '.join(sorted(VALID_BONE_REGIONS))}. "
                        "Bone modeli yalnızca üst ekstremite için eğitilmiştir; "
                        "bacak, diz, ayak, kalça, omurga gibi bölgeler desteklenmez."
                    ),
                )

        if ext not in XRAY_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail="X-ray için desteklenmeyen format",
            )

        contents = await file.read()
        logger.info(
            "analyze file received | user_hash=%s document_type=%s ext=%s size_bytes=%s",
            _hash_user(user_id), document_type, ext, len(contents),
        )

        if len(contents) > MAX_XRAY_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=422, detail=f"X-ray dosyası {MAX_XRAY_SIZE_MB}MB'dan büyük.")

        if not _verify_xray_magic(ext, contents):
            raise HTTPException(
                status_code=422,
                detail="Dosya içeriği uzantıyla uyuşmuyor (magic-byte doğrulaması başarısız).",
            )

        predictor = getattr(request.app.state, "xray_predictor", None)
        if predictor is None:
            logger.error("xray predictor not initialized in app state")
            raise HTTPException(status_code=502, detail="X-ray analizi başarısız")

        logger.info(
            "xray inference started | user_hash=%s xray_type=%s ext=%s",
            _hash_user(user_id), xray_type, ext,
        )

        # CPU inference için eş zamanlılık sınırı (main.py lifespan'da yaratılır)
        sem: asyncio.Semaphore | None = getattr(request.app.state, "xray_sem", None)
        try:
            async with _maybe_acquire(sem):
                if xray_type == "chest":
                    raw_result = await asyncio.to_thread(predictor.predict_chest, contents)
                    analysis = map_chest_to_analysis(raw_result)
                else:
                    raw_result = await asyncio.to_thread(predictor.predict_bone, contents)
                    raw_result["bone_region"] = bone_region
                    raw_result["bone_region_label"] = _BONE_REGION_TR.get(bone_region or "", "")
                    analysis = map_bone_to_analysis(raw_result)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "xray inference failed | user_hash=%s xray_type=%s error=%s",
                _hash_user(user_id), xray_type, exc,
            )
            raise HTTPException(status_code=502, detail="X-ray analizi başarısız")

        inference_ms = raw_result.get("inference_ms")
        logger.info(
            "xray inference done | user_hash=%s xray_type=%s inference_ms=%s severity=%s",
            _hash_user(user_id), xray_type, inference_ms, analysis["severity"],
        )

        # LLM özeti — profil-aware, her zaman çalışır (normal sonuçlarda da).
        # Hata olursa boş string döner, isteği bozmaz.
        try:
            profile = await get_user_profile(user_id)
        except Exception as exc:
            logger.warning("profile fetch failed before xray summary: %s", exc)
            profile = {}

        try:
            ai_summary, ai_summary_error, summary_tokens = await summarize_xray_findings(
                xray_type=xray_type,
                severity=analysis["severity"],
                findings=analysis["findings"],
                raw_inference=raw_result,
                user_profile=profile,
            )
        except RateLimitError:
            logger.warning("rate_limit hit during xray summary | user_hash=%s", _hash_user(user_id))
            ai_summary, ai_summary_error, summary_tokens = "", True, 0

        analysis_id = str(uuid4())
        severity        = analysis["severity"]
        summary         = analysis["summary"]
        findings        = analysis["findings"]
        recommendations = analysis["recommendations"]
        type_specific   = dict(analysis["type_specific"])

        await save_document_analysis(
            user_id=user_id,
            analysis_id=analysis_id,
            document_type=document_type,
            file_name=file.filename,
            severity=severity,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            type_specific=type_specific,
        )


        return {
            "analysis_id":      analysis_id,
            "document_type":    document_type,
            "severity":         severity,
            "summary":          summary,
            "ai_summary":       ai_summary,
            "ai_summary_error": ai_summary_error,
            "findings":         findings,
            "recommendations":  recommendations,
            "type_specific":    type_specific,
            "total_tokens":     summary_tokens,
        }

    # ─── MEVCUT (NON-XRAY) AKIŞ ──────────────────────────────────────────────
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Desteklenmeyen dosya tipi: {ext}")

    contents = await file.read()
    logger.info(
        "analyze file received | user_hash=%s document_type=%s ext=%s size_bytes=%s",
        _hash_user(user_id), document_type, ext, len(contents),
    )

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=422, detail=f"Dosya {MAX_FILE_SIZE_MB}MB'dan büyük.")

    try:
        raw_text = await asyncio.to_thread(parse_file, file.filename, contents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    raw_text = raw_text[:MAX_RAW_TEXT_CHARS]

    try:
        result, analyze_tokens = await analyze_document(raw_text, document_type)
    except RateLimitError:
        logger.warning("rate_limit hit during analyze | user_hash=%s", _hash_user(user_id))
        raise HTTPException(status_code=429, detail="Şu an yoğunluk var, birkaç dakika sonra tekrar dener misin?")

    if not result:
        raise HTTPException(status_code=502, detail="Belge analizi üretilemedi.")

    analysis_id = str(uuid4())
    severity        = result.get("severity", "normal")
    summary         = result.get("summary", "")
    findings        = result.get("findings") or []
    recommendations = result.get("recommendations") or []
    type_specific   = result.get("type_specific") or {}

    await save_document_analysis(
        user_id=user_id,
        analysis_id=analysis_id,
        document_type=document_type,
        file_name=file.filename,
        severity=severity,
        summary=summary,
        findings=findings,
        recommendations=recommendations,
        type_specific=type_specific,
    )




    # Mobil DTO (AiAnalyzeResponse) ai_summary ve ai_summary_error alanlarını
    # her analiz tipi için bekliyor. Non-xray dalında ayrı bir LLM özetlemesi
    # yapmadığımız için summary'yi ai_summary olarak da geçiriyoruz (degraded
    # değil, çünkü asıl analiz başarılı). Bu sayede mobil tek bir code path
    # ile her iki dalı handle eder.
    return {
        "analysis_id":      analysis_id,
        "document_type":    document_type,
        "severity":         severity,
        "summary":          summary,
        "ai_summary":       summary,
        "ai_summary_error": False,
        "findings":         findings,
        "recommendations":  recommendations,
        "type_specific":    type_specific,
        "total_tokens":     analyze_tokens,
    }


# ─── /analyses ───────────────────────────────────────────────────────────────

@router.get("/analyses")
@limiter.limit(CHAT_RATE_LIMIT)
async def list_analyses(
    request: Request,
    user_id: str = Depends(validated_user_id_query),
):
    rows = await get_user_document_analyses(user_id)
    return rows


# ─── /chat ───────────────────────────────────────────────────────────────────

@router.post("/blood-sugar/ai-summary")
@limiter.limit(ANALYZE_RATE_LIMIT)
async def blood_sugar_ai_summary(request: Request, payload: BloodSugarAiSummaryPayload):
    user_id = str(payload.user_id)

    # Bu endpoint sadece token tavanına tabi (sohbet/belge sayacına yazılmaz).

    measurements = await get_recent_blood_sugar_measurements(user_id, days=7)
    fallback_summary = _fallback_blood_sugar_summary(measurements)

    profile: dict = {}
    try:
        profile = await get_user_profile(user_id)
    except Exception as exc:
        logger.warning("profile fetch failed before blood sugar summary: %s", exc)

    ai_summary = ""
    tokens = 0
    try:
        ai_summary, tokens = await summarize_blood_sugar(measurements, profile)
    except RateLimitError:
        logger.warning("rate_limit hit during blood sugar summary | user_hash=%s", _hash_user(user_id))
    except Exception as exc:
        logger.warning("blood sugar summary failed | user_hash=%s error=%s", _hash_user(user_id), exc)




    return {
        "summary":      ai_summary or fallback_summary,
        "total_tokens": tokens,
    }


# ─── /blood-pressure/analyze ─────────────────────────────────────────────────
# Tek bir tansiyon ölçümünü (sistolik / diastolik / opsiyonel nabız) profil +
# geçmiş analizler + RAG ile birlikte yorumlar. Çıktı, Java tarafındaki
# BloodPressureAnalyzeResponse DTO'su ile birebir uyumlu camelCase JSON'dur.
# sayaçlarına yazılmaz; ölçüm fonksiyonel olarak chat'ten farklı bir akıştır).

@router.post("/blood-pressure/analyze")
@limiter.limit(ANALYZE_RATE_LIMIT)
async def blood_pressure_analyze(request: Request, payload: BloodPressureAnalyzePayload):
    user_id = str(payload.user_id)
    systolic = payload.systolic
    diastolic = payload.diastolic
    heart_rate = payload.heart_rate

    # Sistolik en az diastolik + 10 olmalı — aksi halde ölçüm hatası vardır;
    # LLM'i bunun üstüne göndermek yerine baştan kullanıcıyı uyar.
    if systolic <= diastolic:
        raise HTTPException(
            status_code=422,
            detail="Sistolik değer diastolik değerden büyük olmalı.",
        )


    profile: dict = {}
    try:
        profile = await get_user_profile(user_id)
    except Exception as exc:
        logger.warning("profile fetch failed before blood pressure analyze: %s", exc)

    try:
        recent_analyses = await get_recent_analyses(user_id, limit=3)
    except Exception as exc:
        logger.warning("recent analyses fetch failed before blood pressure analyze: %s", exc)
        recent_analyses = []

    # RAG: tansiyon-ilişkili tıbbi bilgi (hipertansiyon, hipertansif kriz)
    rag_context = ""
    try:
        rag_store: RAGStore = get_rag_store()
        query = f"tansiyon {systolic}/{diastolic} hipertansiyon değerlendirme"
        rag_results = await asyncio.to_thread(rag_store.search, query, 2)
        rag_context = _format_rag_results(rag_results)
    except Exception as exc:
        logger.warning("rag search failed for blood pressure: %s", exc)

    analysis: dict
    tokens = 0
    try:
        analysis, tokens = await analyze_blood_pressure(
            systolic=systolic,
            diastolic=diastolic,
            heart_rate=heart_rate,
            user_profile=profile,
            recent_analyses=recent_analyses,
            rag_context=rag_context,
        )
    except RateLimitError:
        logger.warning(
            "rate_limit hit during blood pressure analyze | user_hash=%s",
            _hash_user(user_id),
        )
        raise HTTPException(
            status_code=429,
            detail="Şu an yoğunluk var, birkaç dakika sonra tekrar dener misin?",
        )
    except Exception as exc:
        logger.exception(
            "blood pressure analyze failed | user_hash=%s error=%s",
            _hash_user(user_id), exc,
        )
        raise HTTPException(status_code=502, detail="Tansiyon analizi üretilemedi.")



    logger.info(
        "blood pressure analyze done | user_hash=%s systolic=%s diastolic=%s hr=%s "
        "status=%s category=%s urgency=%s tokens=%d",
        _hash_user(user_id), systolic, diastolic, heart_rate,
        analysis.get("status"), analysis.get("category"),
        analysis.get("urgencyLevel"), tokens,
    )

    return {
        **analysis,
        "total_tokens": tokens,
    }


def _format_rag_results(results: list[SearchResult]) -> str:
    """RAG sonuçlarını LLM context bloğu için string'e çevirir."""
    if not results:
        return ""
    return "\n\n".join(
        f"[{r.metadata.get('source', r.metadata.get('doc_id', ''))}]\n{r.text}"
        for r in results
    )


@router.post("/chat")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_endpoint(
    request: Request,
    session_id: str  = Form(..., description="Oturum kimliği"),
    user_id: str     = Depends(validated_user_id_form),
    message: str     = Form(..., description="Kullanıcı mesajı"),
    analysis_id: str | None = Form(None, description="Bağlam olarak eklenecek analiz UUID'si (opsiyonel)"),
    rag_store: RAGStore = Depends(get_rag_store),
):
    session_id = session_id.strip()

    if not session_id:
        raise HTTPException(status_code=422, detail="session_id boş olamaz")

    if not message.strip():
        raise HTTPException(status_code=422, detail="message boş olamaz")

    message = message.strip()

    memory = get_memory(session_id)
    is_new_session = not memory.get() and not memory.user_id
    if not memory.user_id:
        memory.user_id = user_id


    profile, history = await asyncio.gather(
        get_user_profile(user_id),
        get_recent_analyses(user_id, limit=3),
    )

    # analysis_id verilmişse ve session'da henüz dosya bağlamı yoksa yükle
    if analysis_id and not memory.has_file():
        analysis = await get_document_analysis(analysis_id, user_id)
        if analysis is None:
            raise HTTPException(status_code=404, detail="Analiz bulunamadı veya bu kullanıcıya ait değil.")
        context_text = _format_analysis_for_context(analysis)
        memory.set_file_raw(context_text, file_name=f"analiz_{analysis_id[:8]}.txt")
        logger.info(
            "analysis context loaded | session_id=%s user_hash=%s analysis_id=%s",
            session_id, _hash_user(user_id), analysis_id,
        )

    memory.add("user", message)

    rag_n = 3 if memory.has_file() else 2
    rag_results = await asyncio.to_thread(rag_store.search, message, rag_n)
    rag_context = _format_rag_results(rag_results)

    file_context, file_tokens = await _resolve_file_context(memory, message)

    try:
        result = await chat(
            messages=memory.get(),
            rag_context=rag_context,
            user_profile=profile,
            file_text=file_context,
            previous_analyses=history,
        )
    except RateLimitError:
        logger.warning("rate_limit hit | session_id=%s user_hash=%s", session_id, _hash_user(user_id))
        
        return {
            "response":     "Şu an yoğunluk var, birkaç dakika sonra tekrar dener misin?",
            "session_id":   session_id,
            "total_tokens": file_tokens,
        }

    memory.add("assistant", result.mesaj)

    # Kayıt politikası:
    #  - Sadece tibbi=True VE tani_arayisi=True olan turlar tabloya yazılır.
    #  - Aynı oturumda art arda gelen aynı semptom+tanı kombinasyonu da yutulur.
    if result.tibbi and result.tani_arayisi:
        record = {
            "semptomlar": result.semptomlar,
            "olasi_taniler": result.olasi_taniler,
            "aksiyon": result.aksiyon,
            "tani_arayisi": result.tani_arayisi,
        }
        prior = memory.get_session_records()
        is_in_session_dup = bool(
            prior
            and prior[-1].get("semptomlar") == record["semptomlar"]
            and prior[-1].get("olasi_taniler") == record["olasi_taniler"]
        )
        if not is_in_session_dup:
            memory.add_analysis_record(record)
            await save_analysis(
                user_id,
                result.semptomlar,
                result.olasi_taniler,
                result.aksiyon,
            )


    #  - Yeni session ise +1 chat_session_start (0 token)
    #  - +1 chat_message kaydı (chat + file_summary tokenları toplam olarak yazılır)
    total_tokens = int(result.total_tokens or 0) + int(file_tokens or 0)

    return {
        "response":     result.mesaj,
        "session_id":   session_id,
        "total_tokens": total_tokens,
    }


# ─── /chat/{session_id} DELETE ───────────────────────────────────────────────

@router.delete("/chat/{session_id}")
@limiter.limit(CLEAR_RATE_LIMIT)
async def clear_chat(request: Request, session_id: str, background_tasks: BackgroundTasks):
    memory = get_memory(session_id)
    user_id = memory.user_id
    records = deepcopy(memory.get_session_records())

    if records and user_id:
        background_tasks.add_task(_finalize_cleared_session, session_id, user_id, records)

    memory.clear()
    return {"status": "cleared", "diagnosis": {}}
