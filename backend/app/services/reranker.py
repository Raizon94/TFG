"""
backend/reranker.py — Cross-Encoder Re-ranking para el pipeline RAG
====================================================================
Singleton lazy que carga un modelo Cross-Encoder de sentence-transformers
y expone una función `rerank()` para reordenar listas de candidatos RAG.

Por defecto usa un modelo multilingüe compacto. Se puede sobreescribir
configurando la variable de entorno RERANKER_MODEL.

Uso:
    from backend.reranker import rerank

    scored = rerank(query="cuarzo rosa amor", docs=["texto1...", "texto2..."])
    # → [(0.92, "texto1..."), (0.31, "texto2...")] ordenados desc por score
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("uvicorn.error")

# ── Config ────────────────────────────────────────────────────────
# Modelo por defecto: multilingual MiniLM Cross-Encoder
# (soporta español, rápido, ~80 MB).
# Para mayor precisión en español se puede usar:
#   'BAAI/bge-reranker-base'  (multilingual, mejor calidad, ~280 MB)
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANKER_MODEL = os.getenv("RERANKER_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

# ── Singleton ────────────────────────────────────────────────────
_reranker = None
_reranker_available = True  # se pone a False si falla la importación


def _get_reranker():
    """Carga el Cross-Encoder (lazy, una sola vez)."""
    global _reranker, _reranker_available

    if not _reranker_available:
        return None

    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            logger.info("Re-ranker: cargando modelo '%s'…", _RERANKER_MODEL)
            _reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)
            logger.info("Re-ranker: modelo cargado OK")
        except Exception as exc:
            logger.warning(
                "Re-ranker no disponible (%s). "
                "Se usará el orden original de ChromaDB.",
                exc,
            )
            _reranker_available = False
            return None

    return _reranker


# ── API pública ──────────────────────────────────────────────────

def rerank(
    query: str,
    docs: list[str],
    top_k: int | None = None,
) -> list[tuple[float, str]]:
    """
    Re-ordena una lista de documentos según la relevancia para una query.

    Args:
        query:  Consulta del usuario.
        docs:   Lista de textos candidatos (chunks de ChromaDB).
        top_k:  Si se especifica, devuelve solo los top_k mejores.
                Si es None, devuelve todos ordenados.

    Returns:
        Lista de (score, doc_text) ordenada de mayor a menor score.
        Si el re-ranker no está disponible, devuelve [(0.0, doc)] en el
        orden original (degradación sin errores).
    """
    if not docs:
        return []

    reranker = _get_reranker()
    if reranker is None:
        # Fallback: devolver en orden original con score=0
        result = [(0.0, d) for d in docs]
        return result[:top_k] if top_k else result

    try:
        pairs: list[tuple[str, str]] = [(query, d) for d in docs]
        scores: list[float] = reranker.predict(pairs).tolist()
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return scored[:top_k] if top_k else scored
    except Exception as exc:
        logger.warning("Re-ranker predict failed: %s — usando orden original", exc)
        result = [(0.0, d) for d in docs]
        return result[:top_k] if top_k else result


def rerank_with_meta(
    query: str,
    entries: list[tuple[str, Any]],
    top_k: int | None = None,
) -> list[tuple[float, str, Any]]:
    """
    Igual que `rerank()` pero preservando metadata asociada a cada doc.

    Args:
        query:   Consulta del usuario.
        entries: Lista de (doc_text, metadata) donde metadata es cualquier
                 objeto asociado al chunk.
        top_k:   Si se especifica, devuelve solo los top_k mejores.

    Returns:
        Lista de (score, doc_text, metadata) ordenada de mayor a menor.
    """
    if not entries:
        return []

    docs = [e[0] for e in entries]
    metas = [e[1] for e in entries]

    scored_docs = rerank(query, docs, top_k=None)

    # Map text → metadata (texto del chunk es suficientemente único)
    meta_lookup: dict[str, Any] = {}
    for doc, meta in zip(docs, metas):
        meta_lookup[doc] = meta

    result = [
        (score, doc, meta_lookup.get(doc))
        for score, doc in scored_docs
    ]
    return result[:top_k] if top_k else result
