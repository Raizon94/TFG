"""
backend/tools.py — Herramientas (tools) del agente LangGraph
=============================================================
Cada función decorada con @tool queda disponible para el LLM.
Se agrupan aquí para mantener agent.py limpio y centrado en la API pública.
"""

from __future__ import annotations

import json as _json
import logging
import math as _math
import re as _re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import alog, get_chroma_collection, get_llm, logger
from app.db.database import (
    PsImage,
    PsOrders,
    PsOrderDetail,
    PsOrderHistory,
    PsOrderStateLang,
    PsProduct,
    PsProductLang,
    SessionLocal,
)

# ── Logger dedicado de personalización ────────────────────────────
rlog = logging.getLogger("recom.trace")
rlog.setLevel(logging.DEBUG)


# ═══════════════════════════════════════════════════════════════════
# Helpers internos
# ═══════════════════════════════════════════════════════════════════

def _build_prestashop_image_url(id_image: int) -> str:
    """Genera URL de imagen de PrestaShop 1.6."""
    digits = "/".join(str(id_image))
    return f"https://uniartminerales.com/img/p/{digits}/{id_image}-home_default.jpg"


_STOP_WORDS = {
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "en", "con", "por", "para", "y", "o", "que", "a", "al", "es",
    "como", "se", "su", "más", "mas", "hay", "me", "mi", "tipo",
    "teneis", "tienen", "tienes", "tener", "quiero", "busco",
    "the", "of", "and", "in", "with", "for", "to", "is",
}


def _extract_keywords(query: str) -> list[str]:
    """Extrae palabras clave eliminando stop-words y tokens cortos."""
    tokens = _re.findall(r"[a-záéíóúüñ]+", query.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]


# ═══════════════════════════════════════════════════════════════════
# Personalización ligera para resultados de búsqueda
# ═══════════════════════════════════════════════════════════════════

_TYPE_PREF_WINDOW = 20
_TYPE_PERSONALIZED_SHARE = 0.5


def _normalize_minmax(values: dict[int, float], neutral: float = 0.5) -> dict[int, float]:
    """Normaliza un diccionario de scores a [0, 1]."""
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    span = max_v - min_v
    if span <= 0:
        return {k: neutral for k in values}
    return {k: (v - min_v) / span for k, v in values.items()}


def _compute_recent_type_affinity(db: Session, id_customer: int) -> dict[str, float]:
    """Calcula afinidad por tipo de producto según compras recientes del cliente.

    Toma las últimas líneas de pedido válidas, pondera por recencia y devuelve
    una distribución normalizada por tipo (ej: {"abalorios": 0.6, "decoracion": 0.4}).
    """
    rows = db.execute(
        text(
            """
            SELECT attr.type, o.date_add
            FROM ps_order_detail od
            JOIN ps_orders o ON od.id_order = o.id_order
            LEFT JOIN ps_ai_product_attr attr ON od.product_id = attr.id_product
            WHERE o.id_customer = :uid AND o.valid = 1
            ORDER BY o.date_add DESC, od.id_order_detail DESC
            LIMIT :lim
            """
        ),
        {"uid": id_customer, "lim": _TYPE_PREF_WINDOW},
    ).fetchall()

    if not rows:
        return {}

    weighted_counts: dict[str, float] = {}
    for idx, row in enumerate(rows):
        raw_type = (row[0] or "").strip().lower()
        if not raw_type:
            continue
        recency_weight = 1.0 / (idx + 1)
        weighted_counts[raw_type] = weighted_counts.get(raw_type, 0.0) + recency_weight

    if not weighted_counts:
        return {}

    total = sum(weighted_counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weighted_counts.items()}


def _build_catalog_rerank_text(
    name: str | None,
    stone: str | None,
    product_type: str | None,
    desc_short: str | None,
) -> str:
    """Construye texto del producto para cross-encoder en catálogo."""
    desc_clean = _re.sub(r"<[^>]+>", "", (desc_short or "")).strip()
    if len(desc_clean) > 260:
        desc_clean = desc_clean[:260] + "…"
    parts = [
        (name or "").strip(),
        f"mineral: {(stone or '').strip()}" if stone else "",
        f"tipo: {(product_type or '').strip()}" if product_type else "",
        desc_clean,
    ]
    return " | ".join(p for p in parts if p)


# ═══════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════


@tool
def search_catalog(
    stone: str | None = None,
    product_type: str | None = None,
    keyword: str | None = None,
    id_customer: int | None = None,
    limit: int = 15,
) -> str:
    """Busca productos en el catálogo filtrando por mineral y formato.

    El LLM debe construir los filtros a partir de la conversación con el
    cliente. Al menos uno de stone, product_type o keyword debe indicarse.

    Los resultados se ordenan por un score híbrido:
    - relevancia semántica de query (cross-encoder re-ranker),
    - popularidad/boost de negocio,
    - afinidad por tipo de compras recientes del cliente (si aplica).

    Si el cliente no especifica product_type, se aplica una mezcla por cuotas
    de tipos afines + resultados globales para mantener exploración.

    Args:
        stone: Nombre del mineral/piedra (ej: "amatista", "cuarzo rosa").
               Si no se indica, no se filtra por mineral.
        product_type: Formato del producto (ej: "colgante", "pulsera",
                      "drusa", "esfera", "anillo"). Si no se indica, no se
                      filtra por formato.
        keyword: Palabra clave adicional para buscar en el nombre del
                 producto (ej: "plata", "grande").
        id_customer: ID del cliente (opcional). Si se indica, personaliza
                 el orden con afinidad por tipos de compra recientes.
        limit: Número máximo de resultados (por defecto 15).

    Returns:
        Lista de productos encontrados con nombre, precio e ID, o mensaje
        si no hay resultados.
    """
    # Validar que al menos un filtro esté presente
    if not stone and not product_type and not keyword:
        return (
            "Necesito al menos un filtro para buscar: mineral (stone), "
            "formato (product_type) o palabra clave (keyword). "
            "Pregunta al cliente qué busca."
        )

    db: Session = SessionLocal()
    try:
        # ── Construir query SQL dinámica ──────────────────────────
        # Traemos un pool amplio para re-ordenar en Python
        _POOL_SIZE = limit * 8
        conditions: list[str] = ["p.active = 1"]
        params: dict[str, Any] = {"lang": 3, "lim": _POOL_SIZE}

        # Filtro por mineral (stone) — búsqueda flexible con LIKE
        if stone:
            stone_clean = stone.strip().lower()
            conditions.append("LOWER(attr.stone) LIKE :stone")
            params["stone"] = f"%{stone_clean}%"

        # Filtro por tipo/formato (product_type)
        if product_type:
            type_clean = product_type.strip().lower()
            conditions.append("LOWER(attr.type) LIKE :ptype")
            params["ptype"] = f"%{type_clean}%"

        # Filtro por palabra clave en el nombre
        if keyword:
            kw_clean = keyword.strip().lower()
            conditions.append("LOWER(pl.name) LIKE :kw")
            params["kw"] = f"%{kw_clean}%"

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT p.id_product, pl.name, p.price,
                   pl.description_short, pl.link_rewrite,
                   img.id_image,
                   attr.stone, attr.type,
                   COALESCE(b.boost_factor, 1.0) AS boost,
                   COALESCE(sv.total_sold, 0) AS sales_vol
            FROM ps_product p
            JOIN ps_product_lang pl
              ON p.id_product = pl.id_product AND pl.id_lang = :lang AND pl.id_shop = 1
            LEFT JOIN ps_ai_product_attr attr
              ON p.id_product = attr.id_product
            LEFT JOIN ps_image img
              ON p.id_product = img.id_product AND img.cover = 1
            LEFT JOIN ps_ai_boost b
              ON p.id_product = b.id_product
            LEFT JOIN (
                SELECT od.product_id, SUM(od.product_quantity) AS total_sold
                FROM ps_order_detail od
                JOIN ps_orders o ON od.id_order = o.id_order
                WHERE o.valid = 1
                GROUP BY od.product_id
            ) sv ON p.id_product = sv.product_id
            WHERE {where_clause}
            ORDER BY boost DESC, sales_vol DESC
            LIMIT :lim
        """

        rows = db.execute(text(sql), params).fetchall()

        rlog.info("═" * 72)
        rlog.info(
            "[QUERY] stone=%r  product_type=%r  keyword=%r  "
            "id_customer=%s  limit=%d  pool_size=%d",
            stone, product_type, keyword, id_customer, limit, _POOL_SIZE,
        )
        rlog.info("[SQL] WHERE %s", where_clause)
        rlog.info("[SQL] Pool devuelto: %d filas", len(rows))

        if not rows:
            filters_desc = []
            if stone:
                filters_desc.append(f"mineral='{stone}'")
            if product_type:
                filters_desc.append(f"formato='{product_type}'")
            if keyword:
                filters_desc.append(f"keyword='{keyword}'")
            return (
                f"No se encontraron productos con los filtros: "
                f"{', '.join(filters_desc)}. "
                "Prueba ampliando la búsqueda o con sinónimos."
            )

        # ── Señal de popularidad normalizada por candidato ─────────
        pop_raw: dict[int, float] = {}
        for row in rows:
            pid = int(row[0])
            sales = float(row[9]) if row[9] else 0.0
            boost = float(row[8]) if row[8] else 1.0
            pop_raw[pid] = _math.log1p(sales) * boost
        pop_scores = _normalize_minmax(pop_raw, neutral=0.5)

        rlog.info("── POOL DE CANDIDATOS (SQL) ──")
        for row in rows:
            pid = int(row[0])
            rlog.info(
                "  pid=%-6d  name=%-45s  price=%7.2f  stone=%-18s  type=%-18s  "
                "boost=%.2f  sales=%d  pop_raw=%.3f  pop_norm=%.3f",
                pid, (row[1] or "")[:45], float(row[2]),
                (row[6] or "")[:18], (row[7] or "")[:18],
                float(row[8]) if row[8] else 1.0,
                int(row[9]) if row[9] else 0,
                pop_raw.get(pid, 0.0), pop_scores.get(pid, 0.0),
            )

        # ── Señal de relevancia query→producto (cross-encoder) ─────
        query_parts = [
            f"mineral: {stone.strip()}" if stone else "",
            f"tipo: {product_type.strip()}" if product_type else "",
            f"keyword: {keyword.strip()}" if keyword else "",
        ]
        rerank_query = " | ".join(p for p in query_parts if p)

        rerank_texts = [
            _build_catalog_rerank_text(row[1], row[6], row[7], row[3])
            for row in rows
        ]

        cross_scores: dict[int, float] = {}
        cross_raw: dict[int, float] = {}
        try:
            from app.services.reranker import rerank as _rerank

            ranked = _rerank(rerank_query, rerank_texts, top_k=None)
            doc_indices: dict[str, list[int]] = {}
            for i, doc in enumerate(rerank_texts):
                doc_indices.setdefault(doc, []).append(i)

            used: set[int] = set()
            for score, doc in ranked:
                for idx in doc_indices.get(doc, []):
                    if idx in used:
                        continue
                    used.add(idx)
                    pid = int(rows[idx][0])
                    cross_raw[pid] = float(score)
                    break
            cross_scores = _normalize_minmax(cross_raw, neutral=0.5)
            rlog.info("── CROSS-ENCODER ──")
            rlog.info("  rerank_query=%r", rerank_query)
            for pid in cross_raw:
                rlog.info(
                    "  pid=%-6d  cross_raw=%+.4f  cross_norm=%.3f",
                    pid, cross_raw[pid], cross_scores.get(pid, 0.0),
                )
        except Exception as rerank_exc:
            logger.debug("Catalog reranker skipped: %s", rerank_exc)
            rlog.info("── CROSS-ENCODER: SKIPPED (%s) ──", rerank_exc)

        # ── Afinidad por tipo reciente del cliente (si no fija tipo) ─
        type_affinity: dict[str, float] = {}
        if id_customer and not product_type:
            type_affinity = _compute_recent_type_affinity(db, id_customer)
            rlog.info("── TYPE AFFINITY ──")
            if type_affinity:
                for t, share in sorted(type_affinity.items(), key=lambda x: x[1], reverse=True):
                    rlog.info("  type=%-20s  share=%.3f", t, share)
            else:
                rlog.info("  (sin historial de tipos para customer=%s)", id_customer)
        elif id_customer and product_type:
            rlog.info("── TYPE AFFINITY: DESACTIVADA (product_type=%r fijado) ──", product_type)
        else:
            rlog.info("── TYPE AFFINITY: DESACTIVADA (sin id_customer) ──")

        # ── Score híbrido por producto ─────────────────────────────
        scored_rows: list[tuple[float, Any]] = []
        has_cross = bool(cross_scores)
        has_type_affinity = bool(type_affinity)

        if has_cross and has_type_affinity:
            weights_label = "0.55*cross + 0.30*pop + 0.15*type"
        elif has_cross:
            weights_label = "0.65*cross + 0.35*pop"
        elif has_type_affinity:
            weights_label = "0.75*pop + 0.25*type"
        else:
            weights_label = "1.00*pop"
        rlog.info("── SCORE HÍBRIDO (fórmula: %s) ──", weights_label)

        for row in rows:
            pid = int(row[0])
            row_type = (row[7] or "").strip().lower()

            cross = cross_scores.get(pid, 0.5 if has_cross else 0.0)
            pop = pop_scores.get(pid, 0.5)
            type_score = type_affinity.get(row_type, 0.0) if has_type_affinity else 0.0

            if has_cross and has_type_affinity:
                score = 0.55 * cross + 0.30 * pop + 0.15 * type_score
            elif has_cross:
                score = 0.65 * cross + 0.35 * pop
            elif has_type_affinity:
                score = 0.75 * pop + 0.25 * type_score
            else:
                score = pop

            rlog.info(
                "  pid=%-6d  %-40s  cross=%.3f  pop=%.3f  type=%.3f (%-15s)  → score=%.4f",
                pid, (row[1] or "")[:40], cross, pop, type_score,
                row_type[:15], score,
            )

            scored_rows.append((score, row))

        scored_rows.sort(key=lambda x: x[0], reverse=True)
        rlog.info("── RANKING TRAS SCORE (top %d de %d) ──", limit, len(scored_rows))
        for rank_pos, (sc, rw) in enumerate(scored_rows[:limit], 1):
            rlog.info(
                "  #%-2d  pid=%-6d  score=%.4f  %s",
                rank_pos, int(rw[0]), sc, (rw[1] or "")[:50],
            )

        # ── Mezcla por cuotas de tipo (solo sin product_type explícito) ─
        final_rows = scored_rows[:limit]
        if has_type_affinity and not product_type and scored_rows:
            personalized_slots = min(limit, max(1, int(round(limit * _TYPE_PERSONALIZED_SHARE))))
            raw_quotas = {
                t: personalized_slots * share
                for t, share in type_affinity.items()
            }
            quotas = {t: int(_math.floor(v)) for t, v in raw_quotas.items()}
            remainder = personalized_slots - sum(quotas.values())
            if remainder > 0:
                ordered_remainders = sorted(
                    raw_quotas.items(),
                    key=lambda kv: kv[1] - _math.floor(kv[1]),
                    reverse=True,
                )
                for t, _ in ordered_remainders:
                    if remainder <= 0:
                        break
                    quotas[t] = quotas.get(t, 0) + 1
                    remainder -= 1

            rlog.info("── MEZCLA POR CUOTAS DE TIPO ──")
            rlog.info("  personalized_slots=%d  global_slots=%d", personalized_slots, limit - personalized_slots)
            for t, q in sorted(quotas.items(), key=lambda x: x[1], reverse=True):
                rlog.info("  tipo=%-20s  quota=%d  (raw=%.2f)", t, q, raw_quotas.get(t, 0))

            selected_indices: set[int] = set()
            mixed_rows: list[tuple[float, Any]] = []

            for target_type, quota in quotas.items():
                if quota <= 0:
                    continue
                picked = 0
                for idx, item in enumerate(scored_rows):
                    row_type = (item[1][7] or "").strip().lower()
                    if row_type != target_type or idx in selected_indices:
                        continue
                    mixed_rows.append(item)
                    selected_indices.add(idx)
                    rlog.info(
                        "    [quota %-15s] picked pid=%-6d  score=%.4f  %s",
                        target_type[:15], int(item[1][0]), item[0], (item[1][1] or "")[:40],
                    )
                    picked += 1
                    if picked >= quota:
                        break

            for idx, item in enumerate(scored_rows):
                if len(mixed_rows) >= limit:
                    break
                if idx in selected_indices:
                    continue
                mixed_rows.append(item)
                rlog.info(
                    "    [global         ] filled pid=%-6d  score=%.4f  %s",
                    int(item[1][0]), item[0], (item[1][1] or "")[:40],
                )

            final_rows = mixed_rows[:limit]
        else:
            rlog.info("── MEZCLA POR CUOTAS: NO APLICA ──")

        # ── Formatear los top resultados ──────────────────────────
        results: list[str] = []
        products_data: list[dict] = []

        for _score, row in final_rows:
            pid, name, price, desc_short, link_rw, id_image, \
                r_stone, r_type, boost, sales = row

            desc_clean = ""
            if desc_short:
                desc_clean = _re.sub(r"<[^>]+>", "", desc_short).strip()
                if len(desc_clean) > 150:
                    desc_clean = desc_clean[:150] + "…"

            image_url = _build_prestashop_image_url(id_image) if id_image else ""
            product_link = (
                f"https://uniartminerales.com/{pid}-{link_rw}.html"
                if link_rw else ""
            )

            line = f"• **{name}** — {float(price):.2f} € (ID: {pid})"
            if desc_clean:
                line += f"\n  _{desc_clean}_"
            results.append(line)

            products_data.append({
                "id": pid,
                "name": name,
                "price": float(price),
                "image_url": image_url,
                "link": product_link,
            })

        text_out = (
            f"Se encontraron {len(results)} producto(s):\n"
            + "\n".join(results)
        )
        text_out += (
            f"\n<<PRODUCTS_JSON>>{_json.dumps(products_data)}<</PRODUCTS_JSON>>"
        )

        if has_type_affinity and not product_type:
            personalized = "hybrid+type-mix"
        elif has_cross:
            personalized = "hybrid"
        else:
            personalized = "popularity"

        rlog.info("── RESULTADO FINAL ──")
        for pos, (_sc, rw) in enumerate(final_rows, 1):
            rlog.info(
                "  #%-2d  pid=%-6d  score=%.4f  type=%-15s  stone=%-15s  "
                "price=%.2f  %s",
                pos, int(rw[0]), _sc, (rw[7] or "")[:15],
                (rw[6] or "")[:15], float(rw[2]),
                (rw[1] or "")[:50],
            )
        rlog.info(
            "[RESUMEN] stone=%r type=%r kw=%r customer=%s "
            "mode=%s  pool=%d → final=%d",
            stone, product_type, keyword, id_customer,
            personalized, len(rows), len(final_rows),
        )
        rlog.info("═" * 72)
        return text_out
    finally:
        db.close()


@tool
def get_recommendations(id_customer: int) -> str:
    """Genera recomendaciones personalizadas para un cliente basadas en su
    historial de compras usando IA (Filtrado Colaborativo SVD).

    Args:
        id_customer: ID del cliente en la tienda.

    Returns:
        Lista de productos recomendados o mensaje si no hay datos.
    """
    from app.main import ml_what, ml_fallback
    from app.services import ml_engine

    db: Session = SessionLocal()
    try:
        recommended_ids: list[int] = []

        if ml_what is not None:
            recommended_ids = ml_engine.predict_what(id_customer, db, ml_what)

        if not recommended_ids and ml_fallback:
            recommended_ids = ml_fallback[:6]
            source = "popularidad"
        else:
            source = "IA personalizada"

        if not recommended_ids:
            return (
                "No tengo suficiente información para generar recomendaciones "
                "personalizadas para este cliente. Puedo buscar productos en "
                "el catálogo si me indicas qué tipo de mineral o producto busca."
            )

        rows = (
            db.query(PsProduct, PsProductLang, PsImage)
            .join(PsProductLang, PsProduct.id_product == PsProductLang.id_product)
            .join(PsImage, PsProduct.id_product == PsImage.id_product)
            .filter(
                PsProduct.id_product.in_(recommended_ids),
                PsProductLang.id_lang == 1,
                PsImage.cover == 1,
                PsProduct.active == 1,
            )
            .all()
        )

        # Preservar orden del SVD
        rank = {pid: i for i, pid in enumerate(recommended_ids)}
        products = sorted(
            rows, key=lambda r: rank.get(r[0].id_product, 9999)
        )[:6]

        results = []
        for prod, lang, img in products:
            results.append(
                f"• **{lang.name}** — {float(prod.price):.2f} € "
                f"(ID: {prod.id_product})"
            )

        return (
            f"Recomendaciones ({source}) para el cliente #{id_customer}:\n"
            + "\n".join(results)
        )
    finally:
        db.close()


@tool
def get_order_status(id_customer: int, order_ref: str | None = None) -> str:
    """Consulta el estado de los pedidos de un cliente.

    Args:
        id_customer: ID del cliente.
        order_ref: Referencia del pedido (opcional). Si no se proporciona,
                   devuelve los últimos pedidos del cliente.

    Returns:
        Estado actual del/los pedido(s) con historial.
    """
    db: Session = SessionLocal()
    try:
        query = db.query(PsOrders).filter(
            PsOrders.id_customer == id_customer,
            PsOrders.valid == 1,
        )

        if order_ref:
            query = query.filter(PsOrders.reference == order_ref.upper())

        orders = query.order_by(PsOrders.date_add.desc()).limit(5).all()

        if not orders:
            if order_ref:
                return (
                    f"No se encontró ningún pedido con referencia '{order_ref}' "
                    f"para el cliente #{id_customer}."
                )
            return f"El cliente #{id_customer} no tiene pedidos registrados."

        # Cargar mapa de estados legibles
        state_names: dict[int, str] = {}
        state_rows = (
            db.query(PsOrderStateLang)
            .filter(PsOrderStateLang.id_lang == 1)
            .all()
        )
        for s in state_rows:
            state_names[s.id_order_state] = s.name

        results = []
        for order in orders:
            state_label = state_names.get(
                order.current_state, f"Estado #{order.current_state}"
            )

            # Obtener productos del pedido
            details = (
                db.query(PsOrderDetail)
                .filter(PsOrderDetail.id_order == order.id_order)
                .all()
            )
            product_lines = [
                f"  - {d.product_name} (x{d.product_quantity})" for d in details[:5]
            ]

            # Historial de estados
            history = (
                db.query(PsOrderHistory)
                .filter(PsOrderHistory.id_order == order.id_order)
                .order_by(PsOrderHistory.date_add.desc())
                .limit(3)
                .all()
            )
            history_lines = []
            for h in history:
                h_label = state_names.get(
                    h.id_order_state, f"Estado #{h.id_order_state}"
                )
                history_lines.append(
                    f"  - {h.date_add.strftime('%d/%m/%Y %H:%M')}: {h_label}"
                )

            order_text = (
                f"📦 **Pedido #{order.id_order}** (Ref: {order.reference})\n"
                f"  Estado actual: **{state_label}**\n"
                f"  Fecha: {order.date_add.strftime('%d/%m/%Y')}\n"
                f"  Total: {float(order.total_paid_tax_incl):.2f} €\n"
                f"  Productos:\n" + "\n".join(product_lines)
            )
            if history_lines:
                order_text += "\n  Historial:\n" + "\n".join(history_lines)

            results.append(order_text)

        return "\n\n".join(results)
    finally:
        db.close()


@tool
def search_knowledge_base(question: str) -> str:
    """Busca información en la base de conocimiento de la tienda
    (propiedades de minerales, políticas de envío, devoluciones,
    información general, categorías, etc.) usando búsqueda semántica RAG.

    NO devuelve productos para comprar — solo información/conocimiento.
    Para buscar productos, usa search_catalog con filtros SQL.

    Args:
        question: Pregunta o tema a buscar.

    Returns:
        Información relevante encontrada.
    """
    try:
        col = get_chroma_collection()

        q = (question or "").strip()
        q_l = q.lower()
        query_terms = set(_extract_keywords(q))
        policy_terms = {
            "envio", "envíos", "shipping", "devolucion", "devoluciones",
            "reembolso", "internacional", "internacionales", "entrega",
            "pedido", "politica", "política", "condiciones", "legal",
        }
        is_policy_query = bool(query_terms & policy_terms) or any(
            t in q_l for t in policy_terms
        )

        # 1) Recuperación semántica global
        semantic_results = col.query(
            query_texts=[q],
            n_results=36,
            include=["documents", "metadatas", "distances"],
        )

        # 2) Recuperación dedicada CMS para preguntas de políticas
        cms_results = None
        if is_policy_query:
            cms_results = col.query(
                query_texts=[q],
                n_results=24,
                where={"source": "cms"},
                include=["documents", "metadatas", "distances"],
            )

        # 3) Unificar candidatos y puntuar híbrido
        candidates: list[dict] = []

        def _source_key(meta: dict, doc: str) -> str:
            source = meta.get("source", "?")
            if source == "product_knowledge":
                return f"product_{meta.get('id_product', '?')}"
            if source == "category":
                return f"category_{meta.get('id_category', '?')}"
            if source == "cms":
                return f"cms_{meta.get('id_cms', '?')}"
            return f"other_{hash(doc[:80])}"

        def _source_label(meta: dict) -> str:
            source = meta.get("source", "?")
            if source == "product_knowledge":
                return f"[Mineral: {meta.get('name', '?')}]"
            if source == "category":
                return f"[Categoría: {meta.get('name', '?')}]"
            if source == "cms":
                return f"[Info tienda: {meta.get('title', '?')}]"
            return ""

        def _push(results_obj, from_cms_pass: bool = False):
            if not results_obj or not results_obj.get("documents") or not results_obj["documents"][0]:
                return
            for doc, meta, dist in zip(
                results_obj["documents"][0],
                results_obj["metadatas"][0],
                results_obj["distances"][0],
            ):
                if dist > 1.20:
                    continue
                text_l = (doc or "").lower()
                overlap = 0.0
                if query_terms:
                    overlap = sum(1 for t in query_terms if t in text_l) / len(query_terms)

                semantic_sim = max(0.0, 1.20 - float(dist)) / 1.20
                source = meta.get("source", "?")
                source_prior = 0.08 if source == "cms" else 0.0
                if from_cms_pass:
                    source_prior += 0.06

                hybrid = semantic_sim + 0.55 * overlap + source_prior
                candidates.append(
                    {
                        "doc": doc or "",
                        "meta": meta,
                        "key": _source_key(meta, doc or ""),
                        "label": _source_label(meta),
                        "hybrid": hybrid,
                    }
                )

        _push(semantic_results, from_cms_pass=False)
        _push(cms_results, from_cms_pass=True)

        if not candidates:
            return (
                "No encontré información específica sobre esa consulta en "
                "la base de conocimiento."
            )

        # 4) Preselección por score híbrido y deduplicación
        candidates.sort(key=lambda x: x["hybrid"], reverse=True)
        unique_by_doc: dict[str, dict] = {}
        for c in candidates:
            if c["doc"] not in unique_by_doc:
                unique_by_doc[c["doc"]] = c
        preselected = list(unique_by_doc.values())[:36]

        # 5) Re-rank con cross-encoder
        try:
            from app.services.reranker import rerank as _rerank_kb

            raw_docs = [c["doc"] for c in preselected]
            ranked = _rerank_kb(q, raw_docs, top_k=None)

            rank_score: dict[str, float] = {}
            if ranked:
                min_s = min(float(s) for s, _ in ranked)
                max_s = max(float(s) for s, _ in ranked)
                span = max(max_s - min_s, 1e-6)
                for s, d in ranked:
                    rank_score[d] = (float(s) - min_s) / span

            for c in preselected:
                c["final_score"] = 0.62 * rank_score.get(c["doc"], 0.0) + 0.38 * c["hybrid"]
        except Exception as _re_exc:
            logger.debug("Reranker skipped in search_kb: %s", _re_exc)
            for c in preselected:
                c["final_score"] = c["hybrid"]

        # 6) Selección final con cuotas por fuente
        preselected.sort(key=lambda x: x["final_score"], reverse=True)
        max_total = 12
        max_per_source = 3
        selected: list[str] = []
        seen_source_counts: dict[str, int] = {}

        for c in preselected:
            source_count = seen_source_counts.get(c["key"], 0)
            if source_count >= max_per_source:
                continue
            seen_source_counts[c["key"]] = source_count + 1
            selected.append(f"{c['label']} {c['doc']}")
            if len(selected) >= max_total:
                break

        if not selected:
            return (
                "No encontré información específica sobre esa consulta en "
                "la base de conocimiento."
            )

        alog.debug(
            "  [RAG kb] question=%r policy=%s candidates=%d selected=%d",
            question,
            is_policy_query,
            len(candidates),
            len(selected),
        )
        return "\n\n---\n\n".join(selected)

    except Exception as exc:
        logger.warning("RAG query failed: %s", exc)
        return (
            "No pude consultar la base de conocimiento en este momento. "
            "Intenta de nuevo o pregúntame de otra forma."
        )


@tool
def escalate_to_human(reason: str) -> str:
    """Escala la conversación a un agente humano o al canal de administración.
    Usa esta herramienta cuando:
    - El cliente pide explícitamente hablar con un humano
    - No puedes resolver la consulta con las herramientas disponibles
    - Se trata de una queja o reclamación compleja
    - Hay un problema técnico que no puedes solucionar

    Args:
        reason: Motivo del escalado (breve descripción).

    Returns:
        Confirmación del escalado.
    """
    logger.info("🔀 Escalado a humano — Motivo: %s", reason)
    return (
        f"ESCALADO_CONFIRMADO|{reason}|"
        "La conversación ha sido derivada al equipo de atención al cliente. "
        "Un agente humano se pondrá en contacto contigo lo antes posible."
    )


@tool
def browse_categories(category_name: str | None = None, limit: int = 10) -> str:
    """Explora las categorías de productos de la tienda.
    Si se indica un nombre de categoría, muestra los productos de esa categoría.
    Si no, lista las categorías disponibles.

    Args:
        category_name: Nombre (parcial) de la categoría a explorar.
                       Si es None, lista las categorías principales.
        limit: Número máximo de productos a mostrar por categoría.

    Returns:
        Lista de categorías o productos dentro de la categoría.
    """
    db: Session = SessionLocal()
    try:
        if not category_name:
            # Listar categorías principales con contador de productos
            rows = db.execute(
                text(
                    "SELECT cl.id_category, cl.name, COUNT(cp.id_product) as cnt "
                    "FROM ps_category_lang cl "
                    "JOIN ps_category c ON cl.id_category = c.id_category "
                    "LEFT JOIN ps_category_product cp ON cl.id_category = cp.id_category "
                    "WHERE cl.id_lang = 3 AND c.active = 1 AND c.level_depth >= 2 "
                    "GROUP BY cl.id_category, cl.name "
                    "HAVING cnt > 0 "
                    "ORDER BY cl.name"
                )
            ).fetchall()
            if not rows:
                # Fallback a inglés
                rows = db.execute(
                    text(
                        "SELECT cl.id_category, cl.name, COUNT(cp.id_product) as cnt "
                        "FROM ps_category_lang cl "
                        "JOIN ps_category c ON cl.id_category = c.id_category "
                        "LEFT JOIN ps_category_product cp ON cl.id_category = cp.id_category "
                        "WHERE cl.id_lang = 1 AND c.active = 1 AND c.level_depth >= 2 "
                        "GROUP BY cl.id_category, cl.name "
                        "HAVING cnt > 0 "
                        "ORDER BY cl.name"
                    )
                ).fetchall()

            if not rows:
                return "No se encontraron categorías."

            lines = [f"• **{r[1]}** ({r[2]} productos)" for r in rows]
            return (
                f"Categorías disponibles ({len(lines)}):\n" + "\n".join(lines)
            )
        else:
            # Buscar categoría por nombre y listar sus productos
            keywords = _extract_keywords(category_name)
            if not keywords:
                keywords = [category_name.strip()]

            # Buscar en español primero, luego inglés
            cat_id = None
            cat_display_name = category_name
            for lang_id in [3, 1]:
                for kw in keywords:
                    row = db.execute(
                        text(
                            "SELECT cl.id_category, cl.name "
                            "FROM ps_category_lang cl "
                            "JOIN ps_category c ON cl.id_category = c.id_category "
                            "WHERE cl.id_lang = :lang AND c.active = 1 "
                            "AND cl.name LIKE :pattern LIMIT 1"
                        ),
                        {"lang": lang_id, "pattern": f"%{kw}%"},
                    ).first()
                    if row:
                        cat_id = row[0]
                        cat_display_name = row[1]
                        break
                if cat_id:
                    break

            if not cat_id:
                return f"No se encontró la categoría '{category_name}'."

            # Productos en esa categoría
            prods = db.execute(
                text(
                    "SELECT p.id_product, pl.name, p.price "
                    "FROM ps_category_product cp "
                    "JOIN ps_product p ON cp.id_product = p.id_product "
                    "JOIN ps_product_lang pl ON p.id_product = pl.id_product "
                    "  AND pl.id_lang = 3 "
                    "WHERE cp.id_category = :cat AND p.active = 1 "
                    "ORDER BY p.price DESC LIMIT :lim"
                ),
                {"cat": cat_id, "lim": limit},
            ).fetchall()

            if not prods:
                return (
                    f"La categoría **{cat_display_name}** existe pero "
                    "no tiene productos activos."
                )

            lines = [
                f"• **{r[1]}** — {float(r[2]):.2f} € (ID: {r[0]})" for r in prods
            ]
            return (
                f"Productos en **{cat_display_name}** "
                f"({len(lines)} mostrados):\n" + "\n".join(lines)
            )
    finally:
        db.close()


@tool
def get_customer_info(id_customer: int) -> str:
    """Obtiene información básica del cliente (nombre).
    Usa esta herramienta cuando el cliente pregunte quién es, su nombre,
    o necesites datos del perfil.

    Args:
        id_customer: ID del cliente.

    Returns:
        Nombre y datos básicos del cliente.
    """
    db: Session = SessionLocal()
    try:
        from app.db.database import PsCustomer

        customer = (
            db.query(PsCustomer)
            .filter(PsCustomer.id_customer == id_customer)
            .first()
        )
        if not customer:
            return f"No se encontró un cliente con ID #{id_customer}."

        return (
            f"Cliente #{id_customer}: "
            f"{customer.firstname} {customer.lastname} "
            f"(email: {customer.email})"
        )
    finally:
        db.close()


@tool
def infer_minerals_for_intent(intent: str, limit: int = 6) -> str:
    """Descubre qué minerales o piedras recomienda la WEB DE LA TIENDA para
    una intención o uso concreto: amor, protección, dormir, energía, etc.

    Consulta la base de conocimiento (descripciones de productos, categorías,
    páginas CMS) y usa un modelo de lenguaje para extraer exactamente qué
    minerales menciona la tienda para ese uso.

    IMPORTANTE: Úsala ANTES de search_catalog cuando el cliente pregunte
    por uso/efecto, no por un mineral concreto. Devuelve "MINERALS_NOT_FOUND"
    si no encuentra información relevante.

    Args:
        intent: Lo que busca el cliente, ej: "amor", "proteccion casa",
                "dormir mejor", "limpiar energias"
        limit: Máximo de minerales a devolver (por defecto 6)

    Returns:
        Lista de minerales recomendados por la tienda con contexto.
    """
    try:
        col = get_chroma_collection()

        # Buscar en toda la base de conocimiento
        results = col.query(
            query_texts=[intent],
            n_results=30,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return "MINERALS_NOT_FOUND"

        # Filtrar: solo chunks suficientemente relevantes
        seen_keys: dict[str, int] = {}
        MAX_PER_SOURCE = 3
        MAX_TOTAL = 20
        relevant_chunks: list[str] = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist > 0.70:
                continue
            source = meta.get("source", "?")
            if source == "product_knowledge":
                key = f"product_{meta.get('id_product', '?')}"
                label = meta.get("name", "")
            elif source == "category":
                key = f"cat_{meta.get('id_category', '?')}"
                label = meta.get("name", "")
            elif source == "cms":
                key = f"cms_{meta.get('id_cms', '?')}"
                label = meta.get("title", "")
            else:
                continue

            seen_keys[key] = seen_keys.get(key, 0) + 1
            if seen_keys[key] > MAX_PER_SOURCE:
                continue

            relevant_chunks.append(f"[{label}] {doc[:350]}")
            if len(relevant_chunks) >= MAX_TOTAL:
                break

        alog.debug(
            "  [RAG infer] intent=%r  chunks_ok=%d (dist≤0.70)",
            intent,
            len(relevant_chunks),
        )

        if not relevant_chunks:
            return "MINERALS_NOT_FOUND"

        # Re-rank con Cross-Encoder antes del LLM extractor
        try:
            from app.services.reranker import rerank as _rerank_inf

            raw_docs_inf = [
                chunk.split("] ", 1)[1] if "] " in chunk else chunk
                for chunk in relevant_chunks
            ]
            ranked_inf = _rerank_inf(intent, raw_docs_inf, top_k=len(relevant_chunks))
            doc_to_inf_indices: dict[str, list[int]] = {}
            for _i, _d in enumerate(raw_docs_inf):
                doc_to_inf_indices.setdefault(_d, []).append(_i)
            _used_inf: set[int] = set()
            _reranked: list[str] = []
            for _, _d in ranked_inf:
                for _idx in doc_to_inf_indices.get(_d, []):
                    if _idx not in _used_inf:
                        _used_inf.add(_idx)
                        _reranked.append(relevant_chunks[_idx])
                        break
            relevant_chunks = _reranked
        except Exception as _re_exc:
            logger.debug("Reranker skipped in infer_minerals: %s", _re_exc)

        # Mini-LLM: extrae minerales de los textos de la tienda
        context_text = "\n\n".join(relevant_chunks)
        llm = get_llm()
        extraction_messages = [
            SystemMessage(
                content=(
                    "Eres un extractor de información especializado en minerales y cristales. "
                    "Se te dan fragmentos de texto extraídos de una tienda de minerales. "
                    "Tu tarea: identificar TODOS los minerales/piedras/cristales "
                    "que aparecen en los textos ASOCIADOS a la intención del usuario. "
                    "Responde ÚNICAMENTE con un JSON válido, sin markdown, así:\n"
                    '{"minerals": ["nombre1", "nombre2"], "context": "frase breve"}\n'
                    "Reglas:\n"
                    "- Incluye TODOS los minerales que APAREZCAN EXPLÍCITAMENTE en los textos "
                    "y que tengan CUALQUIER relación con la intención del usuario, "
                    "aunque la mención sea breve o secundaria.\n"
                    "- Lee TODOS los fragmentos cuidadosamente; no te centres solo en "
                    "los primeros o los más obvios.\n"
                    "- Máximo 6 minerales, ordenados de mayor a menor relevancia.\n"
                    "- Nombres en español, minúsculas, singular (ej: cuarzo rosa, amatista).\n"
                    "- 'context' debe resumir brevemente qué dice la tienda sobre cada uno "
                    "en relación con la intención.\n"
                    '- Si no hay minerales relevantes: {"minerals": [], "context": ""}'
                )
            ),
            HumanMessage(
                content=(
                    f"Intención del usuario: '{intent}'\n\n"
                    f"Textos de la tienda:\n{context_text}"
                )
            ),
        ]

        response = llm.invoke(extraction_messages)
        raw = response.content.strip()

        # Limpiar markdown fences si el modelo las añade
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:].strip()

        extracted = _json.loads(raw)
        minerals: list[str] = extracted.get("minerals", [])[:limit]
        context: str = extracted.get("context", "")

        if not minerals:
            return "MINERALS_NOT_FOUND"

        lines = [
            f"Minerales que la tienda asocia con '{intent}':",
        ]
        for m in minerals:
            lines.append(f"  - {m}")
        if context:
            lines.append(f"\nContexto según la tienda: {context}")
        lines.append(
            "\nUsa search_catalog con stone=<mineral> para cada uno "
            "(en paralelo). Si el cliente indicó formato, añade product_type."
        )
        logger.info("infer_minerals_for_intent('%s') → %s", intent, minerals)
        alog.info(
            "  [MINERALS] intent=%r → %s",
            intent,
            minerals,
        )
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("infer_minerals_for_intent failed: %s", exc)
        return "MINERALS_NOT_FOUND"


# ═══════════════════════════════════════════════════════════════════
# Lista de herramientas para el grafo
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    infer_minerals_for_intent,
    search_catalog,
    browse_categories,
    get_recommendations,
    get_order_status,
    get_customer_info,
    search_knowledge_base,
    escalate_to_human,
]
