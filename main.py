import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from api.routes import router


# ─── Rate limiter (global) ────────────────────────────────────────────────────
# Anahtarlama: IP. Route içinde key_func override edilerek user_id ile birleştirilir.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

# CPU üzerinde aynı anda kaç xray inference koşsun?
XRAY_INFERENCE_CONCURRENCY = int(os.getenv("ZENVORA_XRAY_CONCURRENCY", "2"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log = logging.getLogger(__name__)

    # Cold-start gecikmesini almamak için embedder'ı ve DB pool'u önceden ısıt
    from core.rag import warmup
    from core.database import get_pool, close_pool

    # DATABASE_URL yoksa stub modda çalışıyoruz; pool kurma denemesi gereksiz
    if os.getenv("DATABASE_URL"):
        try:
            await asyncio.gather(asyncio.to_thread(warmup), get_pool())
        except Exception as exc:
            log.warning("db pool init failed (stub moduna düşülüyor): %s", exc)
    else:
        log.info("DATABASE_URL boş — stub mode, DB pool kurulmadı")
        await asyncio.to_thread(warmup)

    # X-ray predictor'ı startup'ta bir kez yükle (modeller ~1–2 sn).
    # /analyze endpoint'i request.app.state.xray_predictor üzerinden erişir.
    try:
        from core.xray_inference import XRayPredictor
        app.state.xray_predictor = await asyncio.to_thread(XRayPredictor)
        log.info("xray predictor initialized")
    except Exception as exc:
        app.state.xray_predictor = None
        log.error("xray predictor init failed: %s", exc)

    # CPU inference için eş zamanlılık sınırı — PyTorch thread pool'unu koru
    app.state.xray_sem = asyncio.Semaphore(XRAY_INFERENCE_CONCURRENCY)

    yield

    # ─── Shutdown ─────────────────────────────────────────────────────────────
    log.info("lifespan shutdown started")
    try:
        await close_pool()
        log.info("db pool closed")
    except Exception as exc:
        log.warning("db pool close failed: %s", exc)
    # Predictor referansını bırak — GC modeli serbest bıraksın
    app.state.xray_predictor = None
    log.info("lifespan shutdown complete")


app = FastAPI(title="Zenvora API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ─── CORS — env-driven ────────────────────────────────────────────────────────
# ZENVORA_CORS_ORIGINS="https://app.zenvora.com,https://staging.zenvora.com"
# Boşsa hiçbir origin'e izin verilmez (güvenli default).
_cors_raw = os.getenv("ZENVORA_CORS_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

app.include_router(router, prefix="/api")


@app.get("/health")
async def health(request: Request):
    """Gerçek bileşen sağlığı: DB ping, model yüklü mü, RAG embedder hazır mı."""
    from core import database as _db
    from core import rag_embedder as _rag_emb

    # DB ping — pool varsa kısa SELECT 1
    db_ok = False
    if _db._pool is not None:
        try:
            async with _db._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False

    components = {
        "xray_predictor": getattr(request.app.state, "xray_predictor", None) is not None,
        "db":             db_ok,
        "rag_embedder":   _rag_emb._ST_MODEL is not None,
    }
    overall = "ok" if all(components.values()) else "degraded"
    return {"status": overall, "components": components}


if __name__ == "__main__":
    import uvicorn

    # Geliştirme için: ZENVORA_DEV_RELOAD=true ile aç. Prod default: kapalı.
    reload = os.getenv("ZENVORA_DEV_RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=reload)
