"""
backend/admin_tools.py — Herramientas del agente de administración ERP
=======================================================================
Herramientas de solo lectura para consultar ventas, pedidos,
clientes y catálogo. Diseñadas para el asistente del panel admin.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import logger
from app.db.database import (
    PsAiProductAttr,
    PsCustomer,
    PsOrderDetail,
    PsOrderHistory,
    PsOrderStateLang,
    PsOrders,
    PsProduct,
    PsProductLang,
    SessionLocal,
)

# ── Directorio para imágenes locales de productos ───────────────────
_IMAGES_DIR = Path(__file__).resolve().parent.parent / "product_images"
_IMAGES_DIR.mkdir(exist_ok=True)

# Imágenes pendientes: thread_id → {"path": str, "mime": str}
_pending_images: dict[str, dict[str, str]] = {}


def stage_product_image(thread_id: str, image_bytes: bytes, mime_type: str) -> str:
    """Guarda la imagen subida en staging y la asocia al thread_id.

    Returns:
        Ruta del archivo de staging.
    """
    ext_map = {"image/jpeg": ".jpg", "image/jpg": ".jpg",
               "image/png": ".png", "image/webp": ".webp"}
    ext = ext_map.get(mime_type, ".jpg")
    staging_dir = _IMAGES_DIR / "_staging"
    staging_dir.mkdir(exist_ok=True)
    staging_path = staging_dir / f"{thread_id}{ext}"
    staging_path.write_bytes(image_bytes)
    _pending_images[thread_id] = {"path": str(staging_path), "mime": mime_type}
    logger.info("stage_product_image: thread=%s → %s", thread_id[:8], staging_path)
    return str(staging_path)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

_PERIOD_SQL = {
    "hoy":     "DATE(date_add) = CURDATE()",
    "semana":  "date_add >= DATE_SUB(NOW(), INTERVAL 7 DAY)",
    "mes":     "date_add >= DATE_SUB(NOW(), INTERVAL 30 DAY)",
    "ano":     "date_add >= DATE_SUB(NOW(), INTERVAL 365 DAY)",
    "todo":    "1=1",
}

def _period_clause(period: str) -> str:
    """Devuelve la cláusula SQL para el período solicitado."""
    return _PERIOD_SQL.get(period.lower(), _PERIOD_SQL["mes"])


def _state_names(db: Session) -> dict[int, str]:
    """Carga el mapa id_state → nombre legible (español)."""
    rows = (
        db.query(PsOrderStateLang)
        .filter(PsOrderStateLang.id_lang == 3)
        .all()
    )
    if not rows:
        rows = (
            db.query(PsOrderStateLang)
            .filter(PsOrderStateLang.id_lang == 1)
            .all()
        )
    return {r.id_order_state: r.name for r in rows}


# ═══════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════


@tool
def get_sales_report(period: str = "mes") -> str:
    """Genera un resumen de ventas para el período indicado.

    Args:
        period: "hoy", "semana", "mes", "ano" o "todo"

    Returns:
        Resumen con número de pedidos, ingresos totales y ticket medio.
    """
    clause = _period_clause(period)
    db: Session = SessionLocal()
    try:
        row = db.execute(text(
            f"SELECT COUNT(*) AS cnt, "
            f"       COALESCE(SUM(total_paid_tax_incl), 0) AS revenue, "
            f"       COALESCE(AVG(total_paid_tax_incl), 0) AS avg_ticket "
            f"FROM ps_orders "
            f"WHERE valid = 1 AND {clause}"
        )).fetchone()

        cnt = int(row[0])
        revenue = float(row[1])
        avg_ticket = float(row[2])

        # Desglose por estado
        state_rows = db.execute(text(
            f"SELECT o.current_state, sl.name, COUNT(*) AS c "
            f"FROM ps_orders o "
            f"LEFT JOIN ps_order_state_lang sl "
            f"  ON o.current_state = sl.id_order_state AND sl.id_lang = 3 "
            f"WHERE o.valid = 1 AND o.{clause} "
            f"GROUP BY o.current_state, sl.name "
            f"ORDER BY c DESC LIMIT 8"
        )).fetchall()

        lines = [
            f"📊 **Informe de ventas — {period}**",
            f"• Pedidos: **{cnt:,}**",
            f"• Ingresos totales: **{revenue:,.2f} €**",
            f"• Ticket medio: **{avg_ticket:.2f} €**",
        ]
        if state_rows:
            lines.append("\n**Por estado:**")
            for row in state_rows:
                label = row[1] or f"Estado #{row[0]}"
                lines.append(f"  - {label}: {int(row[2])}")

        return "\n".join(lines)
    except Exception as exc:
        logger.warning("get_sales_report failed: %s", exc)
        return f"No pude generar el informe de ventas: {exc}"
    finally:
        db.close()


@tool
def get_top_products(limit: int = 10, period: str = "mes") -> str:
    """Lista los productos más vendidos en el período indicado.

    Args:
        limit: Número de productos a mostrar (por defecto 10).
        period: "hoy", "semana", "mes", "ano" o "todo"

    Returns:
        Ranking de productos con unidades vendidas e ingresos.
    """
    clause = _period_clause(period)
    db: Session = SessionLocal()
    try:
        rows = db.execute(text(
            f"SELECT od.product_id, od.product_name, "
            f"       SUM(od.product_quantity) AS units, "
            f"       SUM(od.total_price_tax_incl) AS revenue "
            f"FROM ps_order_detail od "
            f"JOIN ps_orders o ON od.id_order = o.id_order "
            f"WHERE o.valid = 1 AND o.{clause} "
            f"GROUP BY od.product_id, od.product_name "
            f"ORDER BY units DESC "
            f"LIMIT :lim"
        ), {"lim": limit}).fetchall()

        if not rows:
            return f"No hay datos de ventas para el período '{period}'."

        lines = [f"🏆 **Top {limit} productos más vendidos — {period}**"]
        for i, row in enumerate(rows, 1):
            lines.append(
                f"{i}. **{row[1]}** — {int(row[2])} uds "
                f"({float(row[3]):,.2f} €)"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("get_top_products failed: %s", exc)
        return f"No pude obtener el ranking de productos: {exc}"
    finally:
        db.close()


@tool
def search_orders(
    customer_name: str | None = None,
    reference: str | None = None,
    status: str | None = None,
    limit: int = 10,
) -> str:
    """Busca pedidos por nombre de cliente, referencia o estado.

    Args:
        customer_name: Nombre o apellido del cliente (parcial, opcional).
        reference: Referencia del pedido, ej: "XYZAB12" (opcional).
        status: Texto del estado, ej: "pendiente", "entregado" (opcional).
        limit: Máximo de resultados (por defecto 10).

    Returns:
        Lista de pedidos con detalles del cliente, estado y total.
    """
    db: Session = SessionLocal()
    try:
        # Construir query dinámica
        conditions = ["o.valid = 1"]
        params: dict = {"lim": limit}

        joins = [
            "JOIN ps_customer c ON o.id_customer = c.id_customer",
        ]

        if customer_name:
            conditions.append(
                "(c.firstname LIKE :cname OR c.lastname LIKE :cname "
                " OR CONCAT(c.firstname, ' ', c.lastname) LIKE :cname)"
            )
            params["cname"] = f"%{customer_name}%"

        if reference:
            conditions.append("o.reference LIKE :ref")
            params["ref"] = f"%{reference.upper()}%"

        if status:
            joins.append(
                "JOIN ps_order_state_lang sl "
                "  ON o.current_state = sl.id_order_state AND sl.id_lang = 3"
            )
            conditions.append("sl.name LIKE :status")
            params["status"] = f"%{status}%"
        else:
            joins.append(
                "LEFT JOIN ps_order_state_lang sl "
                "  ON o.current_state = sl.id_order_state AND sl.id_lang = 3"
            )

        where = " AND ".join(conditions)
        join_sql = "\n".join(joins)

        rows = db.execute(text(
            f"SELECT o.id_order, o.reference, "
            f"       c.firstname, c.lastname, c.email, "
            f"       sl.name AS state_name, "
            f"       o.total_paid_tax_incl, o.date_add "
            f"FROM ps_orders o "
            f"{join_sql} "
            f"WHERE {where} "
            f"ORDER BY o.date_add DESC "
            f"LIMIT :lim"
        ), params).fetchall()

        if not rows:
            return "No se encontraron pedidos con los filtros indicados."

        lines = [f"🔎 **{len(rows)} pedido(s) encontrado(s):**"]
        for r in rows:
            state = r[5] or f"Estado #{r[6]}"
            date_str = r[7].strftime("%d/%m/%Y") if r[7] else "—"
            lines.append(
                f"\n📦 **Pedido #{r[0]}** (Ref: {r[1]})\n"
                f"  Cliente: {r[2]} {r[3]} ({r[4]})\n"
                f"  Estado: {state} · {float(r[6]):.2f} € · {date_str}"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("search_orders failed: %s", exc)
        return f"No pude buscar los pedidos: {exc}"
    finally:
        db.close()


@tool
def get_order_details(order_id: int) -> str:
    """Obtiene todos los detalles de un pedido: líneas, cliente, historial.

    Args:
        order_id: ID numérico del pedido.

    Returns:
        Información completa del pedido.
    """
    db: Session = SessionLocal()
    try:
        order = db.query(PsOrders).filter(PsOrders.id_order == order_id).first()
        if not order:
            return f"No se encontró ningún pedido con ID #{order_id}."

        states = _state_names(db)
        state_label = states.get(order.current_state, f"Estado #{order.current_state}")

        # Datos del cliente
        customer = (
            db.query(PsCustomer)
            .filter(PsCustomer.id_customer == order.id_customer)
            .first()
        )
        customer_str = (
            f"{customer.firstname} {customer.lastname} ({customer.email})"
            if customer
            else f"ID #{order.id_customer}"
        )

        # Dirección de envío
        try:
            addr = db.execute(text(
                "SELECT a.firstname, a.lastname, a.address1, "
                "       COALESCE(a.address2, '') AS address2, "
                "       COALESCE(a.postcode, '') AS postcode, "
                "       a.city, "
                "       COALESCE(co.name, '') AS country "
                "FROM ps_address a "
                "LEFT JOIN ps_country_lang co "
                "  ON a.id_country = co.id_country AND co.id_lang = 3 "
                "WHERE a.id_address = :ia AND a.deleted = 0"
            ), {"ia": order.id_address_delivery}).fetchone()
        except Exception:
            addr = None

        if addr:
            parts = [f"{addr[0]} {addr[1]}", addr[2]]
            if addr[3]:
                parts.append(addr[3])
            parts.append(f"{addr[4]} {addr[5]}".strip())
            if addr[6]:
                parts.append(addr[6])
            addr_str = ", ".join(p for p in parts if p)
        else:
            addr_str = "No disponible"

        # Líneas del pedido
        details = (
            db.query(PsOrderDetail)
            .filter(PsOrderDetail.id_order == order_id)
            .all()
        )
        product_lines = [
            f"  - {d.product_name} × {d.product_quantity} "
            f"({float(d.total_price_tax_incl):.2f} €)"
            for d in details
        ]

        # Historial de estados
        history = (
            db.query(PsOrderHistory)
            .filter(PsOrderHistory.id_order == order_id)
            .order_by(PsOrderHistory.date_add.asc())
            .all()
        )
        history_lines = [
            f"  - {h.date_add.strftime('%d/%m/%Y %H:%M')}: "
            f"{states.get(h.id_order_state, f'Estado #{h.id_order_state}')}"
            for h in history
        ]

        lines = [
            f"📦 **Pedido #{order.id_order}** (Ref: {order.reference or '—'})",
            f"  Cliente: {customer_str}",
            f"  Estado: **{state_label}**",
            f"  Fecha: {order.date_add.strftime('%d/%m/%Y %H:%M')}",
            f"  Total: {float(order.total_paid_tax_incl):.2f} €",
            f"  Pago: {order.payment}",
            f"  Dirección de envío: {addr_str}",
            "\n  **Productos:**",
            *product_lines,
        ]
        if history_lines:
            lines += ["\n  **Historial:**", *history_lines]

        return "\n".join(lines)
    except Exception as exc:
        logger.warning("get_order_details failed: %s", exc)
        return f"No pude obtener los detalles del pedido #{order_id}: {exc}"
    finally:
        db.close()


@tool
def search_customers(query: str, limit: int = 10) -> str:
    """Busca clientes por nombre, apellido o email.

    Args:
        query: Texto a buscar (nombre, apellido o email, parcial).
        limit: Máximo de resultados (por defecto 10).

    Returns:
        Lista de clientes con sus datos básicos y número de pedidos.
    """
    db: Session = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT c.id_customer, c.firstname, c.lastname, c.email, "
            "       c.active, COUNT(o.id_order) AS num_orders, "
            "       COALESCE(SUM(o.total_paid_tax_incl), 0) AS total_spent "
            "FROM ps_customer c "
            "LEFT JOIN ps_orders o "
            "  ON c.id_customer = o.id_customer AND o.valid = 1 "
            "WHERE c.deleted = 0 "
            "  AND (c.firstname LIKE :q OR c.lastname LIKE :q OR c.email LIKE :q "
            "       OR CONCAT(c.firstname, ' ', c.lastname) LIKE :q) "
            "GROUP BY c.id_customer, c.firstname, c.lastname, c.email, c.active "
            "ORDER BY num_orders DESC "
            "LIMIT :lim"
        ), {"q": f"%{query}%", "lim": limit}).fetchall()

        if not rows:
            return f"No se encontraron clientes para '{query}'."

        lines = [f"👥 **{len(rows)} cliente(s) encontrado(s):**"]
        for r in rows:
            active_str = "✅ activo" if r[4] else "❌ inactivo"
            lines.append(
                f"\n👤 **{r[1]} {r[2]}** ({r[3]})\n"
                f"   {active_str} · {int(r[5])} pedidos · "
                f"{float(r[6]):,.2f} € total"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("search_customers failed: %s", exc)
        return f"No pude buscar clientes: {exc}"
    finally:
        db.close()


@tool
def get_catalog_stats() -> str:
    """Obtiene estadísticas generales del catálogo de productos.

    Returns:
        Número de productos activos/inactivos, precio medio, y
        las categorías con más productos.
    """
    db: Session = SessionLocal()
    try:
        totals = db.execute(text(
            "SELECT "
            "  COUNT(*) AS total, "
            "  SUM(active) AS activos, "
            "  COUNT(*) - SUM(active) AS inactivos, "
            "  AVG(price) AS precio_medio, "
            "  MIN(price) AS precio_min, "
            "  MAX(price) AS precio_max "
            "FROM ps_product"
        )).fetchone()

        cat_rows = db.execute(text(
            "SELECT cl.name, COUNT(cp.id_product) AS cnt "
            "FROM ps_category_lang cl "
            "JOIN ps_category c ON cl.id_category = c.id_category "
            "LEFT JOIN ps_category_product cp ON cl.id_category = cp.id_category "
            "LEFT JOIN ps_product p ON cp.id_product = p.id_product AND p.active = 1 "
            "WHERE cl.id_lang = 3 AND c.active = 1 AND c.level_depth >= 2 "
            "GROUP BY cl.id_category, cl.name "
            "HAVING cnt > 0 "
            "ORDER BY cnt DESC LIMIT 8"
        )).fetchall()

        lines = [
            "📦 **Estadísticas del catálogo**",
            f"• Total productos: **{int(totals[0]):,}**",
            f"• Activos: **{int(totals[1]):,}** · Inactivos: **{int(totals[2]):,}**",
            f"• Precio medio: **{float(totals[3]):.2f} €**",
            f"• Rango de precios: {float(totals[4]):.2f} € — {float(totals[5]):.2f} €",
        ]

        if cat_rows:
            lines.append("\n**Top categorías (productos activos):**")
            for r in cat_rows:
                lines.append(f"  - {r[0]}: {int(r[1])} productos")

        return "\n".join(lines)
    except Exception as exc:
        logger.warning("get_catalog_stats failed: %s", exc)
        return f"No pude obtener estadísticas del catálogo: {exc}"
    finally:
        db.close()


@tool
def get_pending_orders(limit: int = 15) -> str:
    """Lista los pedidos en estado 'Pago aceptado' pendientes de empaquetar y enviar.

    Args:
        limit: Máximo de pedidos a mostrar (por defecto 15).

    Returns:
        Lista de pedidos con ID, cliente, fecha, total
        y número de productos/unidades a empaquetar.
    """
    db: Session = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT o.id_order, o.reference, "
            "       c.firstname, c.lastname, "
            "       sl.name AS state_name, "
            "       o.total_paid_tax_incl, o.date_add, "
            "       COALESCE(SUM(od.product_quantity), 0) AS total_units, "
            "       COUNT(od.id_order_detail) AS num_lines "
            "FROM ps_orders o "
            "JOIN ps_customer c ON o.id_customer = c.id_customer "
            "LEFT JOIN ps_order_state_lang sl "
            "  ON o.current_state = sl.id_order_state AND sl.id_lang = 3 "
            "LEFT JOIN ps_order_detail od ON o.id_order = od.id_order "
            "WHERE o.valid = 1 AND o.current_state = 2 "
            "GROUP BY o.id_order, o.reference, c.firstname, c.lastname, "
            "         sl.name, o.total_paid_tax_incl, o.date_add "
            "ORDER BY o.date_add ASC "
            "LIMIT :lim"
        ), {"lim": limit}).fetchall()

        if not rows:
            return "No hay pedidos con 'Pago aceptado' pendientes de enviar. 🎉"

        lines = [f"⏳ **{len(rows)} pedido(s) pendiente(s):**"]
        for r in rows:
            state      = r[4] or "—"
            date_str   = r[6].strftime("%d/%m/%Y %H:%M") if r[6] else "—"
            total_units = int(r[7])
            num_lines   = int(r[8])
            lines.append(
                f"  📦 **#{r[0]}** (Ref: {r[1]}) · {r[2]} {r[3]} · "
                f"{state} · {float(r[5]):.2f} € · {date_str} · "
                f"{num_lines} producto(s), {total_units} ud(s)"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("get_pending_orders failed: %s", exc)
        return f"No pude obtener los pedidos pendientes: {exc}"
    finally:
        db.close()


@tool
def get_packing_list(order_id: int) -> str:
    """Devuelve la lista de empaquetado completa de un pedido concreto:
    todos los productos a empaquetar y la dirección de envío.
    Usar después de get_pending_orders para obtener el ID del pedido.

    Args:
        order_id: ID numérico del pedido (columna id_order).

    Returns:
        Detalle completo: productos con nombre, referencia y cantidad,
        más la dirección de envío completa.
    """
    db: Session = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT o.id_order, o.reference, "
            "       c.firstname, c.lastname, "
            "       sl.name AS state_name, "
            "       o.total_paid_tax_incl, o.date_add, "
            "       o.id_address_delivery "
            "FROM ps_orders o "
            "JOIN ps_customer c ON o.id_customer = c.id_customer "
            "LEFT JOIN ps_order_state_lang sl "
            "  ON o.current_state = sl.id_order_state AND sl.id_lang = 3 "
            "WHERE o.id_order = :oid"
        ), {"oid": order_id}).fetchone()

        if not row:
            return f"No se encontró ningún pedido con ID #{order_id}."

        reference = row[1] or "—"
        cust_name = f"{row[2]} {row[3]}"
        state     = row[4] or "—"
        total     = float(row[5])
        date_str  = row[6].strftime("%d/%m/%Y %H:%M") if row[6] else "—"
        id_addr   = row[7]

        # Dirección de envío
        try:
            addr = db.execute(text(
                "SELECT a.firstname, a.lastname, a.address1, "
                "       COALESCE(a.address2, '') AS address2, "
                "       COALESCE(a.postcode, '') AS postcode, "
                "       a.city, "
                "       COALESCE(co.name, '') AS country "
                "FROM ps_address a "
                "LEFT JOIN ps_country_lang co "
                "  ON a.id_country = co.id_country AND co.id_lang = 3 "
                "WHERE a.id_address = :ia AND a.deleted = 0"
            ), {"ia": id_addr}).fetchone()
        except Exception:
            addr = None

        if addr:
            parts = [f"{addr[0]} {addr[1]}", addr[2]]
            if addr[3]:
                parts.append(addr[3])
            parts.append(f"{addr[4]} {addr[5]}".strip())
            if addr[6]:
                parts.append(addr[6])
            addr_str = ", ".join(p for p in parts if p)
        else:
            addr_str = "Dirección no disponible"

        # Productos
        products = db.execute(text(
            "SELECT product_name, product_quantity, product_reference "
            "FROM ps_order_detail "
            "WHERE id_order = :oid "
            "ORDER BY id_order_detail"
        ), {"oid": order_id}).fetchall()

        if not products:
            return f"El pedido #{order_id} no tiene líneas de productos registradas."

        prod_lines = [
            f"  • {p[0]}"
            + (f" (Ref: {p[2]})" if p[2] else "")
            + f" × **{int(p[1])}**"
            for p in products
        ]

        return (
            f"📋 **Lista de empaquetado — Pedido #{order_id}** (Ref: {reference})\n"
            f"👤 Cliente: {cust_name}\n"
            f"📅 Fecha: {date_str} · Estado: {state}\n"
            f"💰 Total: {total:.2f} €\n\n"
            f"📍 **Dirección de envío:**\n  {addr_str}\n\n"
            f"🛍️ **Productos a empaquetar ({len(products)} línea(s)):**\n"
            + "\n".join(prod_lines)
        )
    except Exception as exc:
        logger.warning("get_packing_list failed: %s", exc)
        return f"No pude generar la lista de empaquetado para el pedido #{order_id}: {exc}"
    finally:
        db.close()


@tool
def update_order_status(order_id: int, new_status: str) -> str:
    """Cambia el estado de un pedido existente.

    El parámetro new_status debe ser uno de los estados válidos del sistema.
    Usa list_order_statuses para ver todos los estados disponibles.

    Args:
        order_id: ID numérico del pedido.
        new_status: Nombre exacto del nuevo estado (tal como aparece en list_order_statuses).

    Returns:
        Confirmación del cambio o mensaje de error.
    """
    db: Session = SessionLocal()
    try:
        # ── 1. Verificar que el pedido existe ─────────────────────────
        order = db.query(PsOrders).filter(PsOrders.id_order == order_id).first()
        if not order:
            return f"No se encontró ningún pedido con ID #{order_id}."

        # ── 2. Resolver el estado destino ─────────────────────────────
        state_row = (
            db.query(PsOrderStateLang)
            .filter(
                PsOrderStateLang.id_lang == 3,
                PsOrderStateLang.name == new_status,
            )
            .first()
        )
        if not state_row:
            # Intentar búsqueda case-insensitive con LIKE exacto
            state_row = db.execute(text(
                "SELECT id_order_state, name "
                "FROM ps_order_state_lang "
                "WHERE id_lang = 3 AND LOWER(name) = LOWER(:ns) "
                "LIMIT 1"
            ), {"ns": new_status}).fetchone()
            if not state_row:
                # Devolver la lista completa para que el LLM pueda elegir
                all_states = (
                    db.query(PsOrderStateLang)
                    .filter(PsOrderStateLang.id_lang == 3)
                    .order_by(PsOrderStateLang.id_order_state)
                    .all()
                )
                names = ", ".join(f'"{s.name}"' for s in all_states)
                return (
                    f"Estado '{new_status}' no reconocido.\n"
                    f"Estados válidos: {names}"
                )
            new_state_id = int(state_row[0])
            new_state_name = state_row[1]
        else:
            new_state_id = state_row.id_order_state
            new_state_name = state_row.name

        # ── 3. Verificar que no sea el mismo estado ───────────────────
        old_state_id = order.current_state
        if old_state_id == new_state_id:
            return (
                f"El pedido #{order_id} ya se encuentra en el estado "
                f"'{new_state_name}'. No se realizó ningún cambio."
            )

        # Nombre del estado anterior
        old_state = (
            db.query(PsOrderStateLang)
            .filter(
                PsOrderStateLang.id_order_state == old_state_id,
                PsOrderStateLang.id_lang == 3,
            )
            .first()
        )
        old_state_name = old_state.name if old_state else f"Estado #{old_state_id}"

        # ── 4. Actualizar current_state en ps_orders ──────────────────
        order.current_state = new_state_id
        order.date_upd = datetime.now()

        # ── 5. Insertar entrada en ps_order_history ───────────────────
        db.execute(text(
            "INSERT INTO ps_order_history "
            "(id_employee, id_order, id_order_state, date_add) "
            "VALUES (:emp, :oid, :sid, NOW())"
        ), {"emp": 1, "oid": order_id, "sid": new_state_id})

        db.commit()

        logger.info(
            "update_order_status: pedido #%d  %s → %s",
            order_id, old_state_name, new_state_name,
        )

        return (
            f"✅ Pedido #{order_id} actualizado correctamente.\n"
            f"  Estado anterior: **{old_state_name}**\n"
            f"  Nuevo estado: **{new_state_name}**"
        )
    except Exception as exc:
        db.rollback()
        logger.warning("update_order_status failed: %s", exc)
        return f"No pude actualizar el estado del pedido #{order_id}: {exc}"
    finally:
        db.close()


@tool
def list_order_statuses() -> str:
    """Lista todos los estados de pedido disponibles en el sistema.

    Returns:
        Lista de estados con su ID y nombre.
    """
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(PsOrderStateLang)
            .filter(PsOrderStateLang.id_lang == 3)
            .order_by(PsOrderStateLang.id_order_state)
            .all()
        )
        if not rows:
            return "No se encontraron estados de pedido en la base de datos."

        lines = [f"📋 **{len(rows)} estados de pedido disponibles:**"]
        for r in rows:
            lines.append(f"  {r.id_order_state:>2}. {r.name}")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("list_order_statuses failed: %s", exc)
        return f"No pude obtener los estados de pedido: {exc}"
    finally:
        db.close()


@tool
def create_product(
    name: str,
    description_short: str,
    description: str,
    price: float,
    product_type: str,
    stone: str,
    category_id: int,
    reference: str = "",
    quantity: int = 1,
    active: bool = True,
) -> str:
    """Crea un nuevo producto en la base de datos de PrestaShop.

    Inserta en ps_product, ps_product_lang, ps_product_shop,
    ps_stock_available, ps_category_product y ps_ai_product_attr.

    Args:
        name: Nombre del producto (ej: "Colgante de Amatista natural").
        description_short: Descripción corta HTML (2-3 frases comerciales).
        description: Descripción larga HTML completa del producto.
        price: Precio del producto en euros (sin IVA, decimal).
        product_type: Tipo del producto (debe coincidir con los tipos válidos del catálogo).
        stone: Piedra/mineral principal (debe coincidir con las piedras válidas del catálogo).
        category_id: ID numérico de la categoría de PrestaShop.
        reference: Referencia interna del producto (opcional).
        quantity: Stock inicial (por defecto 1).
        active: Si el producto está activo y visible (por defecto True).

    Returns:
        Confirmación con el ID del nuevo producto o mensaje de error.
    """
    import re as _re
    db: Session = SessionLocal()
    try:
        now = datetime.now()

        # ── 1. Generar link_rewrite a partir del nombre ──────────
        link_rewrite = name.lower().strip()
        link_rewrite = link_rewrite.replace("á", "a").replace("é", "e")
        link_rewrite = link_rewrite.replace("í", "i").replace("ó", "o")
        link_rewrite = link_rewrite.replace("ú", "u").replace("ñ", "n")
        link_rewrite = link_rewrite.replace("ü", "u")
        link_rewrite = _re.sub(r"[^a-z0-9]+", "-", link_rewrite).strip("-")
        if not link_rewrite:
            link_rewrite = "producto"

        # ── 2. Insertar en ps_product ────────────────────────────
        result = db.execute(text(
            "INSERT INTO ps_product "
            "(id_shop_default, id_manufacturer, id_supplier, id_category_default, "
            " id_tax_rules_group, "
            " reference, price, active, quantity, "
            " date_add, date_upd) "
            "VALUES "
            "(1, 0, 0, :cat_id, "
            " 1, "
            " :ref, :price, :active, :qty, "
            " :now, :now)"
        ), {
            "cat_id": category_id,
            "ref": reference,
            "price": price,
            "active": 1 if active else 0,
            "qty": quantity,
            "now": now,
        })
        new_id = result.lastrowid

        # ── 3. Insertar en ps_product_lang (español id_lang=3) ───
        db.execute(text(
            "INSERT INTO ps_product_lang "
            "(id_product, id_shop, id_lang, name, description_short, "
            " description, link_rewrite) "
            "VALUES (:pid, 1, 3, :name, :dshort, :desc, :link)"
        ), {
            "pid": new_id,
            "name": name[:128],
            "dshort": description_short,
            "desc": description,
            "link": link_rewrite[:128],
        })

        # ── 4. Insertar en ps_product_shop ───────────────────────
        db.execute(text(
            "INSERT INTO ps_product_shop "
            "(id_product, id_shop, id_category_default, id_tax_rules_group, "
            " price, active, visibility, date_add, date_upd, "
            " available_for_order, show_price, `condition`, indexed) "
            "VALUES (:pid, 1, :cat_id, 1, "
            " :price, :active, 'both', :now, :now, "
            " 1, 1, 'new', 1)"
        ), {
            "pid": new_id,
            "cat_id": category_id,
            "price": price,
            "active": 1 if active else 0,
            "now": now,
        })

        # ── 5. Insertar en ps_category_product ───────────────────
        # Posición = max actual + 1
        max_pos = db.execute(text(
            "SELECT COALESCE(MAX(position), 0) FROM ps_category_product "
            "WHERE id_category = :cid"
        ), {"cid": category_id}).scalar()

        db.execute(text(
            "INSERT INTO ps_category_product "
            "(id_category, id_product, position) "
            "VALUES (:cid, :pid, :pos)"
        ), {"cid": category_id, "pid": new_id, "pos": max_pos + 1})

        # ── 6. Insertar en ps_stock_available ────────────────────
        db.execute(text(
            "INSERT INTO ps_stock_available "
            "(id_product, id_product_attribute, id_shop, id_shop_group, "
            " quantity, physical_quantity, depends_on_stock, out_of_stock) "
            "VALUES (:pid, 0, 1, 1, :qty, :qty, 0, 2)"
        ), {"pid": new_id, "qty": quantity})

        # ── 7. Insertar en ps_ai_product_attr ────────────────────
        db.execute(text(
            "INSERT INTO ps_ai_product_attr "
            "(id_product, stone, type, updated_at) "
            "VALUES (:pid, :stone, :ptype, NOW())"
        ), {"pid": new_id, "stone": stone, "ptype": product_type})

        # ── 8. Guardar imagen local + ps_image (si hay staging) ───
        image_saved = False
        # Buscar imagen staged para cualquier thread activo
        staged = None
        staged_thread = None
        for tid, info in list(_pending_images.items()):
            if os.path.exists(info["path"]):
                staged = info
                staged_thread = tid
                break

        if staged:
            # Insertar registro en ps_image
            img_result = db.execute(text(
                "INSERT INTO ps_image (id_product, position, cover) "
                "VALUES (:pid, 1, 1)"
            ), {"pid": new_id})
            id_image = img_result.lastrowid

            # Mover archivo de staging a ubicación definitiva
            src = Path(staged["path"])
            ext = src.suffix
            dest = _IMAGES_DIR / f"{id_image}{ext}"
            shutil.move(str(src), str(dest))
            image_saved = True

            # Limpiar staging
            _pending_images.pop(staged_thread, None)
            logger.info(
                "create_product: imagen #%d guardada en %s",
                id_image, dest,
            )

        db.commit()

        logger.info(
            "create_product: nuevo producto #%d '%s' (type=%s, stone=%s, cat=%d, price=%.2f)",
            new_id, name, product_type, stone, category_id, price,
        )

        return (
            f"✅ Producto creado correctamente.\n"
            f"  **ID:** {new_id}\n"
            f"  **Nombre:** {name}\n"
            f"  **Tipo:** {product_type}\n"
            f"  **Piedra:** {stone}\n"
            f"  **Categoría:** {category_id}\n"
            f"  **Precio:** {price:.2f} €\n"
            f"  **Stock:** {quantity}\n"
            f"  **Referencia:** {reference or '—'}\n"
            f"  **Estado:** {'Activo ✅' if active else 'Inactivo ❌'}"
        )
    except Exception as exc:
        db.rollback()
        logger.error("create_product failed: %s", exc, exc_info=True)
        return f"❌ No pude crear el producto: {exc}"
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════
# Lista de herramientas del agente admin
# ═══════════════════════════════════════════════════════════════════

ADMIN_TOOLS = [
    get_sales_report,
    get_top_products,
    search_orders,
    get_order_details,
    search_customers,
    get_catalog_stats,
    get_pending_orders,
    get_packing_list,
    update_order_status,
    list_order_statuses,
    create_product,
]
