"""
backend/agent.py — API pública del agente de cliente
=====================================================
Punto de entrada para main.py y scripts de evaluación.
Funciones públicas: chat(), chat_stream(), get_chat_history().

Toda la lógica pesada está repartida en:
  • config.py   → LLM, ChromaDB, loggers, proxy, checkpointer
  • prompts.py  → SYSTEM_PROMPT
  • tools.py    → 8 herramientas @tool + helpers
  • graph.py    → AgentState, nodos, grafo LangGraph
"""

from __future__ import annotations

import json as _json
import re as _re
import uuid
from typing import Any, Generator

from langchain_core.messages import AIMessage, HumanMessage
from openai import APIConnectionError

from app.core.config import alog, logger
from .graph import get_graph

# ── Re-exports para compatibilidad con evaluate_rag.py ────────────
from app.agents.customer.tools import (  # noqa: F401
    infer_minerals_for_intent,
    search_catalog,
    search_knowledge_base,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers de extracción de marcadores
# ═══════════════════════════════════════════════════════════════════

_PRODUCTS_JSON_RE = _re.compile(
    r"<<PRODUCTS_JSON>>(.*?)<<\s*/\s*PRODUCTS_JSON>>", _re.DOTALL
)
_SHOW_PRODUCTS_RE = _re.compile(
    r"<<SHOW_PRODUCTS>>(.*?)(?:<<\s*/\s*SHOW_PRODUCTS>>|<<SHOW_PRODUCTS>>)",
    _re.DOTALL,
)


def _extract_all_products(messages) -> dict[int, dict]:
    """Recorre los mensajes y extrae el catálogo completo de PRODUCTS_JSON."""
    all_products: dict[int, dict] = {}
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        if "<<PRODUCTS_JSON>>" not in content:
            continue
        for match in _PRODUCTS_JSON_RE.finditer(content):
            try:
                for p in _json.loads(match.group(1)):
                    all_products[p["id"]] = p
            except _json.JSONDecodeError:
                continue
    return all_products


def _extract_selected_ids(text: str) -> list[int]:
    """Extrae los IDs de producto del marcador SHOW_PRODUCTS."""
    match = _SHOW_PRODUCTS_RE.search(text)
    if not match:
        return []
    return [
        int(x.strip())
        for x in match.group(1).split(",")
        if x.strip().isdigit()
    ]


def _clean_markers(text: str) -> str:
    """Elimina los marcadores internos del texto de respuesta."""
    text = _re.sub(
        r"<<SHOW_PRODUCTS>>.*?(?:<<\s*/\s*SHOW_PRODUCTS>>|<<SHOW_PRODUCTS>>)",
        "",
        text,
        flags=_re.DOTALL,
    ).strip()
    text = _re.sub(
        r"<<PRODUCTS_JSON>>.*?<<\s*/\s*PRODUCTS_JSON>>",
        "",
        text,
        flags=_re.DOTALL,
    ).strip()
    # Algunos modelos emiten un prefijo "final" suelto antes de la respuesta
    text = _re.sub(r"^final\s*", "", text, flags=_re.IGNORECASE).strip()
    return text


def _build_tool_status_detail(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Construye un detalle legible para el status de streaming."""
    if tool_name == "search_catalog":
        parts = []
        stone = (tool_args.get("stone") or "").strip()
        product_type = (tool_args.get("product_type") or "").strip()
        keyword = (tool_args.get("keyword") or "").strip()

        if stone:
            parts.append(stone)
        if product_type:
            parts.append(product_type)
        if keyword:
            parts.append(keyword)

        return " | ".join(parts)

    field_by_tool = {
        "search_knowledge_base": "question",
        "infer_minerals_for_intent": "intent",
    }
    field = field_by_tool.get(tool_name)
    if not field:
        return ""

    value = tool_args.get(field)
    return value.strip() if isinstance(value, str) else ""


def _is_expected_connection_error(exc: Exception) -> bool:
    """Detecta errores esperables de red/proxy con el proveedor LLM."""
    if isinstance(exc, APIConnectionError):
        return True
    err_str = str(exc).lower()
    return any(
        token in err_str
        for token in ("connection error", "network is unreachable", "connecterror")
    )


# ═══════════════════════════════════════════════════════════════════
# get_chat_history
# ═══════════════════════════════════════════════════════════════════

def get_chat_history(thread_id: str) -> dict[str, Any]:
    """Recupera el historial de conversación persistido para un thread_id."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        logger.warning("No se pudo leer historial de thread %s: %s", thread_id, exc)
        return {"thread_id": thread_id, "messages": []}

    if not snapshot or not snapshot.values:
        return {"thread_id": thread_id, "messages": []}

    all_messages = snapshot.values.get("messages", []) or []
    if not all_messages:
        return {"thread_id": thread_id, "messages": []}

    all_products = _extract_all_products(all_messages)

    history: list[dict[str, Any]] = []
    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            text = (msg.content or "").strip()
            if text:
                history.append({"role": "user", "content": text})
            continue

        if not isinstance(msg, AIMessage):
            continue

        content = (msg.content or "").strip()
        if not content:
            continue

        selected_ids = _extract_selected_ids(content)
        products = [all_products[pid] for pid in selected_ids if pid in all_products]

        content = _clean_markers(content)
        if not content:
            continue

        history.append({
            "role": "assistant",
            "content": content,
            "products": products,
        })

    return {
        "thread_id": thread_id,
        "messages": history,
    }


def clear_chat_history(thread_id: str) -> bool:
    """Borra el estado de un hilo para empezar de cero."""
    try:
        from app.core.config import memory  # noqa: WPS433
        # SqliteSaver usa un conn sqlite3; borramos las filas de ese thread
        conn = getattr(memory, "conn", None) or getattr(memory, "_conn", None)
        if conn is None:
            logger.warning("No se pudo obtener conn del checkpointer")
            return False
        cur = conn.cursor()
        cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        logger.info("Historial borrado para thread %s", thread_id)
        return True
    except Exception as exc:
        logger.warning("Error al borrar historial de thread %s: %s", thread_id, exc)
        return False


# ═══════════════════════════════════════════════════════════════════
# chat() — invocación síncrona
# ═══════════════════════════════════════════════════════════════════

def chat(
    message: str,
    id_customer: int | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    Punto de entrada principal del agente de cliente.

    Args:
        message: Mensaje del usuario.
        id_customer: ID del cliente (opcional, para personalización).
        thread_id: ID del hilo de conversación (para memoria).

    Returns:
        dict con keys: reply, thread_id, escalated, products
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    alog.info("")
    alog.info("═" * 60)
    alog.info("[ENTRADA] thread=%s customer=%s", thread_id[:8], id_customer)
    alog.info("[USER] %r", message)
    alog.info("-" * 60)

    graph = get_graph()

    input_state = {
        "messages": [HumanMessage(content=message)],
        "id_customer": id_customer,
        "escalated": False,
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(input_state, config=config)

        # Extraer la última respuesta del asistente
        reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        if not reply:
            reply = (
                "Disculpa, no pude procesar tu consulta en este momento. "
                "¿Puedes reformularla?"
            )

        # Extraer productos y limpiar marcadores
        all_products = _extract_all_products(result["messages"])
        selected_ids = _extract_selected_ids(reply)
        unique_products = [
            all_products[pid] for pid in selected_ids if pid in all_products
        ]
        reply = _clean_markers(reply)

        escalated = result.get("escalated", False)

        alog.info("-" * 60)
        alog.info(
            "[RESPUESTA] %d productos seleccionados  escalated=%s",
            len(unique_products),
            escalated,
        )
        alog.info("[REPLY] %s", reply[:300].replace("\n", " "))
        if unique_products:
            names = ", ".join(p["name"] for p in unique_products[:8])
            alog.info("[PRODUCTOS] %s", names)
        alog.info("═" * 60)

        return {
            "reply": reply,
            "thread_id": thread_id,
            "escalated": escalated,
            "products": unique_products,
        }

    except Exception as exc:
        if _is_expected_connection_error(exc):
            logger.warning("LLM no disponible en chat(): %s", exc)
        else:
            logger.error("Error en el agente de cliente: %s", exc, exc_info=True)
        alog.error("[ERROR] %s: %s", type(exc).__name__, exc)
        return {
            "reply": (
                f"⚠️ Error técnico en el agente:\n\n"
                f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                f"Revisa la configuración del LLM o contacta con soporte."
            ),
            "thread_id": thread_id,
            "escalated": False,
        }


# ═══════════════════════════════════════════════════════════════════
# chat_stream() — streaming SSE para el frontend
# ═══════════════════════════════════════════════════════════════════

# Mapa de nombre de herramienta → emoji + descripción amigable
_TOOL_LABELS: dict[str, str] = {
    "infer_minerals_for_intent": "🪨 Analizando textos de la tienda",
    "search_catalog": "📦 Buscando productos en el catálogo",
    "browse_categories": "📂 Explorando categorías",
    "get_recommendations": "✨ Generando recomendaciones",
    "get_order_status": "📋 Consultando estado de pedido",
    "get_customer_info": "👤 Obteniendo info del cliente",
    "search_knowledge_base": "🔍 Buscando en la base de conocimiento",
    "escalate_to_human": "🙋 Escalando a un humano",
}


def chat_stream(
    message: str,
    id_customer: int | None = None,
    thread_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Versión streaming de chat(). Genera eventos dict:
      {"type": "status", "content": "🔍 Buscando..."}
      {"type": "result", "data": {reply, thread_id, escalated, products}}
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    graph = get_graph()

    input_state = {
        "messages": [HumanMessage(content=message)],
        "id_customer": id_customer,
        "escalated": False,
    }
    config = {"configurable": {"thread_id": thread_id}}

    alog.info("")
    alog.info("═" * 60)
    alog.info("[ENTRADA/STREAM] thread=%s customer=%s", thread_id[:8], id_customer)
    alog.info("[USER] %r", message)
    alog.info("-" * 60)

    yield {"type": "status", "content": "🧠 Pensando..."}

    try:
        # stream_mode="updates" da un dict por nodo ejecutado
        final_state = None
        for event in graph.stream(input_state, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                if node_name == "agent":
                    msgs = state_update.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_name = tc["name"]
                                tool_args = tc.get("args", {})
                                label = _TOOL_LABELS.get(
                                    tool_name,
                                    f"🔧 Ejecutando {tool_name}",
                                )
                                detail = _build_tool_status_detail(tool_name, tool_args)
                                if detail:
                                    label += f': "{detail}"'
                                elif tool_name == "browse_categories":
                                    pid = tool_args.get("parent_id")
                                    if pid:
                                        label += f" (id: {pid})"
                                elif tool_name == "get_order_status":
                                    label += "..."
                                yield {"type": "status", "content": label}
                        elif isinstance(msg, AIMessage) and msg.content:
                            yield {
                                "type": "status",
                                "content": "✍️ Redactando respuesta...",
                            }

                elif node_name == "tools":
                    yield {
                        "type": "status",
                        "content": "⚙️ Procesando resultados...",
                    }

            final_state = state_update

        # ── Reconstruir resultado final desde el checkpoint ────────
        snapshot = graph.get_state(config)
        all_messages = snapshot.values.get("messages", [])

        reply = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        if not reply:
            reply = (
                "Disculpa, no pude procesar tu consulta en este momento. "
                "¿Puedes reformularla?"
            )

        all_products = _extract_all_products(all_messages)
        selected_ids = _extract_selected_ids(reply)
        unique_products = [
            all_products[pid] for pid in selected_ids if pid in all_products
        ]
        reply = _clean_markers(reply)

        escalated = snapshot.values.get("escalated", False)

        alog.info("-" * 60)
        alog.info(
            "[RESPUESTA/STREAM] %d productos  escalated=%s",
            len(unique_products),
            escalated,
        )
        alog.info("[REPLY] %s", reply[:300].replace("\n", " "))
        if unique_products:
            names = ", ".join(p["name"] for p in unique_products[:8])
            alog.info("[PRODUCTOS] %s", names)
        alog.info("═" * 60)

        yield {
            "type": "result",
            "data": {
                "reply": reply,
                "thread_id": thread_id,
                "escalated": escalated,
                "products": unique_products,
            },
        }

    except Exception as exc:
        if _is_expected_connection_error(exc):
            logger.warning("LLM no disponible en chat_stream(): %s", exc)
        else:
            logger.error("Error en chat_stream: %s", exc, exc_info=True)
        alog.error("[ERROR/STREAM] %s: %s", type(exc).__name__, exc)
        yield {
            "type": "result",
            "data": {
                "reply": (
                    f"⚠️ Error técnico en el agente:\n\n"
                    f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                    f"Revisa la configuración del LLM o contacta con soporte."
                ),
                "thread_id": thread_id,
                "escalated": False,
                "products": [],
            },
        }
