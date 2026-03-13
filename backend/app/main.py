from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import warnings
import joblib

from app.api.router import api_router
from app.api.utils import _PRODUCT_IMAGES_DIR
from app.core.config import get_chroma_collection
from app.db.database import ensure_ai_tables

logger = logging.getLogger("uvicorn.error")

_RAG_WARMUP_ON_STARTUP = os.getenv("RAG_WARMUP_ON_STARTUP", "0").strip() in {
    "1", "true", "yes", "on"
}

import os
_MODELS_DIR = Path(os.getenv("MODELS_DIR", str(Path(__file__).resolve().parent / "models")))
ml_what: dict | None = None
ml_fallback: list[int] | None = None


def _normalize_uvicorn_loggers() -> None:
    """Evita logs duplicados cuando Uvicorn añade handlers repetidos."""
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(logger_name)
        seen: set[tuple[type, str | None]] = set()
        unique_handlers = []
        for handler in uv_logger.handlers:
            stream = getattr(handler, "stream", None)
            stream_name = getattr(stream, "name", None)
            key = (type(handler), stream_name)
            if key in seen:
                continue
            seen.add(key)
            unique_handlers.append(handler)
        uv_logger.handlers = unique_handlers
        uv_logger.propagate = False

def _reload_models() -> None:
    global ml_what, ml_fallback

    what_path = _MODELS_DIR / "svd_what.joblib"
    fallback_path = _MODELS_DIR / "fallback_popular.joblib"

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = Warning

    if what_path.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            ml_what = joblib.load(what_path)
        logger.info("Modelo SVD cargado (%d usuarios, %d productos)",
                     len(ml_what["customer_ids"]), len(ml_what["product_ids"]))
    else:
        logger.warning("⚠️  Modelo SVD no encontrado en %s. "
                       "Ejecuta: python -m scripts.train_ml", what_path)

    if fallback_path.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            ml_fallback = joblib.load(fallback_path)
        logger.info("Fallback de productos populares cargado (%d productos)", len(ml_fallback))

@asynccontextmanager
async def lifespan(application: FastAPI):
    _normalize_uvicorn_loggers()
    ensure_ai_tables()
    _reload_models()
    if _RAG_WARMUP_ON_STARTUP:
        try:
            get_chroma_collection()
        except Exception as exc:
            logger.warning("RAG no disponible durante startup: %s", exc)
    yield
    logger.info("Cerrando API — liberando modelos ML.")

app = FastAPI(
    title="TFG — E-commerce Headless IA",
    version="0.1.0",
    description="API gestionada por agentes de IA sobre datos PrestaShop.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PRODUCT_IMAGES_DIR.mkdir(exist_ok=True)
app.mount("/product_images", StaticFiles(directory=str(_PRODUCT_IMAGES_DIR)), name="product_images")

app.include_router(api_router, prefix="/api")
