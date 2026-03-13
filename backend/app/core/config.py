"""
backend/config.py — Configuración compartida del agente
========================================================
ChromaDB, LLM, loggers, proxy y checkpointer de memoria.
Todos los demás módulos del agente importan desde aquí.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time

from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Logger principal (uvicorn) ────────────────────────────────────
logger = logging.getLogger("uvicorn.error")

# ── Logger de traza del agente ─────────────────────────────────────
alog = logging.getLogger("agent.trace")
alog.setLevel(logging.DEBUG)

# ── Proxy (Shadowrocket / VPN local) ─────────────────────────────
_PROXY_URL = os.getenv("PROXY_URL", "").strip()
_IN_DOCKER = os.getenv("IN_DOCKER", "0").strip().lower() in {"1", "true", "yes", "on"}
_ENABLE_DOCKER_PROXY = os.getenv("ENABLE_DOCKER_PROXY", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

if _PROXY_URL and (not _IN_DOCKER or _ENABLE_DOCKER_PROXY):
    os.environ["http_proxy"] = _PROXY_URL
    os.environ["https_proxy"] = _PROXY_URL
    os.environ["HTTP_PROXY"] = _PROXY_URL
    os.environ["HTTPS_PROXY"] = _PROXY_URL
    logger.info("Proxy configurado: %s", _PROXY_URL)
else:
    # Si no hay PROXY_URL en el .env, no usar proxy
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    if _PROXY_URL and _IN_DOCKER and not _ENABLE_DOCKER_PROXY:
        logger.info("Proxy desactivado dentro de Docker (ENABLE_DOCKER_PROXY=0)")


# ═══════════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════════

_MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
_API_KEY = os.getenv("OPENAI_API_KEY", None)


def get_llm() -> ChatOpenAI:
    """Instancia del LLM con la config por defecto."""
    kwargs: dict = {
        "model": _MODEL_NAME,
        "temperature": _TEMPERATURE,
    }
    if _BASE_URL:
        kwargs["base_url"] = _BASE_URL
    if _API_KEY:
        kwargs["api_key"] = _API_KEY
    return ChatOpenAI(**kwargs)


# ═══════════════════════════════════════════════════════════════════
# ChromaDB (RAG)
# ═══════════════════════════════════════════════════════════════════

_CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "rag_data"))
_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
_RAG_COLLECTION = os.getenv("RAG_COLLECTION", "uniart_knowledge")
_chroma_collection = None  # singleton lazy
_rag_init_lock = threading.Lock()


def _ensure_rag_collection(client, embedding_fn) -> None:
    """Garantiza que la colección RAG exista; intenta construirla si falta."""
    try:
        client.get_collection(_RAG_COLLECTION, embedding_function=embedding_fn)
        return
    except Exception:
        pass

    # Single-flight: solo un hilo/proceso del worker inicializa RAG a la vez.
    with _rag_init_lock:
        try:
            client.get_collection(_RAG_COLLECTION, embedding_function=embedding_fn)
            return
        except Exception as exc:
            logger.warning(
                "Colección RAG '%s' no encontrada (%s). Intentando construir índice...",
                _RAG_COLLECTION,
                exc,
            )

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        built_ok = False
        try:
            rag_builder = importlib.import_module("scripts.build_rag")
            rag_builder.build_index(force=False)
            built_ok = True
        except Exception as build_exc:
            logger.warning("No se pudo construir el índice RAG automáticamente: %s", build_exc)

        if built_ok:
            return

        # Fallback para evitar fallos continuos cuando no hay colección.
        client.get_or_create_collection(
            _RAG_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(
            "Se creó una colección RAG vacía ('%s'). Ejecuta scripts/build_rag.py para poblarla.",
            _RAG_COLLECTION,
        )


def get_chroma_collection():
    """Devuelve la colección ChromaDB (singleton)."""
    global _chroma_collection
    if _chroma_collection is None:
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL,
        )
        client = chromadb.PersistentClient(path=_CHROMA_DIR)
        _ensure_rag_collection(client, embedding_fn)
        _chroma_collection = client.get_collection(
            _RAG_COLLECTION,
            embedding_function=embedding_fn,
        )
        # Si la colección se creó vacía mientras se inicializaba, esperar
        # brevemente por si otro request terminó de poblarla.
        if _chroma_collection.count() == 0:
            time.sleep(0.2)
        logger.info("ChromaDB: %d documentos cargados", _chroma_collection.count())
    return _chroma_collection


# ═══════════════════════════════════════════════════════════════════
# Checkpointer (SqliteSaver con fallback a MemorySaver)
# ═══════════════════════════════════════════════════════════════════

try:
    import sqlite3 as _sqlite3

    _sqlite_saver_module = None
    for _module_path in (
        "langgraph.checkpoint.sqlite",
        "langgraph.checkpoint.sqlite3",
    ):
        try:
            _sqlite_saver_module = importlib.import_module(_module_path)
            break
        except ImportError:
            continue
    if _sqlite_saver_module is None:
        raise ImportError("SqliteSaver module not found")
    SqliteSaver = _sqlite_saver_module.SqliteSaver
    _USE_SQLITE = True
except ImportError:
    _USE_SQLITE = False

if _USE_SQLITE:
    _SQLITE_DB_PATH = os.getenv("AGENT_MEMORY_DB", os.path.join(os.path.dirname(__file__), "..", "..", "..", "agent_memory.sqlite"))
    _sqlite_conn = _sqlite3.connect(_SQLITE_DB_PATH, check_same_thread=False)
    memory = SqliteSaver(_sqlite_conn)
else:
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
