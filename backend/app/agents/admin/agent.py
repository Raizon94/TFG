"""
backend/admin_agent.py — Agente de administración ERP con LangGraph
====================================================================
Agente conversacional para el panel de administración.
Responde en lenguaje natural sobre ventas, pedidos, clientes y catálogo.

Funciones públicas: chat(), chat_stream(), get_chat_history()
(mismo contrato que customer_agent.py para facilitar la integración)
"""

from __future__ import annotations

import json as _json
import re as _re
import time
import uuid
from typing import Annotated, Any, Generator, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agents.admin.tools import ADMIN_TOOLS
from app.core.config import alog, get_llm, logger, memory

# ═══════════════════════════════════════════════════════════════════
# System Prompt del administrador
# ═══════════════════════════════════════════════════════════════════

_ADMIN_SYSTEM_PROMPT = """\
Eres el **Asistente ERP de UniArt Minerales**, un agente de inteligencia \
artificial especializado en la gestión interna del e-commerce.

**Personalidad:** Directo, analítico y profesional. Respondes siempre en \
español con información precisa extraída de la base de datos real de la \
tienda. Sin inventar cifras: si no tienes datos, dilo claramente.

**Capacidades:**
1. **Informes de ventas** con `get_sales_report`: pedidos, ingresos y \
   ticket medio por período (hoy / semana / mes / año / todo).
2. **Productos más vendidos** con `get_top_products`: ranking por unidades \
   e ingresos en el período indicado.
3. **Buscar pedidos** con `search_orders`: filtra por nombre de cliente, \
   referencia de pedido o estado.
4. **Detalle de pedido** con `get_order_details`: líneas, cliente, \
   historial de estados completo.
5. **Buscar clientes** con `search_customers`: por nombre, apellido o email.
6. **Estadísticas del catálogo** con `get_catalog_stats`: productos activos, \
   precios, categorías.
7. **Pedidos pendientes** con `get_pending_orders`: lista los pedidos en \
   estado **'Pago aceptado'** (pagados pero aún no enviados), incluyendo \
   el ID, el cliente, el estado y el número de productos/unidades.
8. **Lista de empaquetado** con `get_packing_list(order_id)`: dado el ID \
   de un pedido concreto, devuelve todos los productos a empaquetar \
   (nombre, referencia, cantidad) y la dirección de envío completa. \
   Úsala SOLO cuando el admin solicite explícitamente el detalle de \
   empaquetado de un pedido específico; primero muestra la lista con \
   `get_pending_orders` y espera a que el admin indique el ID.
9. **Cambiar estado de pedido** con `update_order_status(order_id, new_status)`: \
   cambia el estado de un pedido. El parámetro `new_status` debe ser \
   el nombre EXACTO de un estado válido del sistema. Si no conoces los \
   nombres exactos, usa `list_order_statuses` primero para consultarlos. \
   Pide SIEMPRE confirmación al admin antes de ejecutar el cambio \
   (muestra el pedido, estado actual y estado destino).
10. **Listar estados de pedido** con `list_order_statuses()`: muestra todos \
    los estados disponibles en el sistema con su ID y nombre.
11. **Crear producto** con `create_product(...)`: inserta un nuevo producto \
    en la base de datos de PrestaShop. Necesita: name, description_short, \
    description (larga, HTML), price, product_type, stone, category_id. \
    Opcionales: reference, quantity (def 1), active (def True). \
    Cuando recibas datos de análisis de imagen (vía [VISION_ANALYSIS]), \
    usa esa información como base pero SIEMPRE: \
    (a) muestra al admin un resumen de lo detectado y pide confirmación, \
    (b) pide el PRECIO al admin (no lo inventes), \
    (c) enriquece la descripción larga con detalles comerciales atractivos, \
    (d) solo llama a create_product tras confirmación del admin.

**Reglas:**
- SIEMPRE usa las herramientas para obtener datos reales; nunca inventes \
  cifras ni datos de clientes o pedidos.
- Cuando el admin pregunte por un período sin especificar, usa "mes" \
  como valor por defecto.
- Formatea las respuestas con markdown: negritas, listas, emojis moderados.
- Si hay un error al consultar la BD, explícalo claramente y sugiere \
  una alternativa.
- Eres solo para uso INTERNO del equipo de administración. No estás \
  expuesto a clientes.
"""


# ═══════════════════════════════════════════════════════════════════
# Estado del grafo
# ═══════════════════════════════════════════════════════════════════

class AdminState(TypedDict):
    """Estado del grafo conversacional del agente admin."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ═══════════════════════════════════════════════════════════════════
# Grafo LangGraph
# ═══════════════════════════════════════════════════════════════════

def _build_admin_graph():
    """Construye el grafo LangGraph del agente administrador."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ADMIN_TOOLS)

    def agent_node(state: AdminState) -> dict:
        messages = list(state["messages"])

        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=_ADMIN_SYSTEM_PROMPT))
        else:
            messages[0] = SystemMessage(content=_ADMIN_SYSTEM_PROMPT)

        try:
            response = llm_with_tools.invoke(messages)
            if isinstance(response, AIMessage) and response.tool_calls:
                calls_desc = ", ".join(
                    f"{tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc.get('args', {}).items())})"
                    for tc in response.tool_calls
                )
                alog.info("  [ADMIN LLM] → TOOL CALLS: %s", calls_desc)
            elif isinstance(response, AIMessage) and response.content:
                alog.info("  [ADMIN LLM] → RESPUESTA FINAL (%d chars)", len(response.content))
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                logger.warning("[admin] Rate-limited — waiting 3s…")
                time.sleep(3)
                try:
                    response = llm_with_tools.invoke(messages)
                except Exception as retry_exc:
                    response = AIMessage(
                        content=f"⚠️ Servicio temporalmente saturado. Inténtalo en unos segundos.\n\n"
                                f"```\n{type(retry_exc).__name__}: {retry_exc}\n```"
                    )
            else:
                raise

        return {"messages": [response]}

    tool_node = ToolNode(ADMIN_TOOLS)

    def should_continue(state: AdminState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AdminState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=memory)


_compiled_admin_graph = None


def _get_graph():
    global _compiled_admin_graph
    if _compiled_admin_graph is None:
        _compiled_admin_graph = _build_admin_graph()
    return _compiled_admin_graph


# ═══════════════════════════════════════════════════════════════════
# API pública
# ═══════════════════════════════════════════════════════════════════

def get_chat_history(thread_id: str) -> dict[str, Any]:
    """Recupera el historial persistido de una sesión admin."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        logger.warning("admin history read failed thread=%s: %s", thread_id, exc)
        return {"thread_id": thread_id, "messages": []}

    if not snapshot or not snapshot.values:
        return {"thread_id": thread_id, "messages": []}

    history: list[dict] = []
    for msg in snapshot.values.get("messages", []) or []:
        if isinstance(msg, HumanMessage):
            text = (msg.content or "").strip()
            if text:
                history.append({"role": "user", "content": text})
        elif isinstance(msg, AIMessage):
            content = (msg.content or "").strip()
            if content:
                history.append({"role": "assistant", "content": content})

    return {"thread_id": thread_id, "messages": history}


def chat(
    message: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Invocación síncrona del agente admin."""
    if not thread_id:
        thread_id = str(uuid.uuid4())

    alog.info("═" * 60)
    alog.info("[ADMIN ENTRADA] thread=%s", thread_id[:8])
    alog.info("[ADMIN USER] %r", message)

    graph = _get_graph()
    input_state: AdminState = {
        "messages": [HumanMessage(content=message)],
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(input_state, config=config)

        reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        if not reply:
            reply = "Disculpa, no pude procesar tu consulta. ¿Puedes reformularla?"

        alog.info("[ADMIN REPLY] %s", reply[:200].replace("\n", " "))
        alog.info("═" * 60)

        return {"reply": reply, "thread_id": thread_id}
    except Exception as exc:
        logger.error("Error en agente admin: %s", exc, exc_info=True)
        return {
            "reply": (
                f"⚠️ Error técnico en el agente ERP:\n\n"
                f"```\n{type(exc).__name__}: {exc}\n```"
            ),
            "thread_id": thread_id,
        }


# Etiquetas amigables para los eventos de streaming
_TOOL_LABELS: dict[str, str] = {
    "get_sales_report":      "📊 Generando informe de ventas",
    "get_top_products":      "🏆 Calculando productos más vendidos",
    "search_orders":         "🔎 Buscando pedidos",
    "get_order_details":     "📦 Cargando detalle del pedido",
    "search_customers":      "👥 Buscando clientes",
    "get_catalog_stats":     "📦 Consultando catálogo",
    "get_pending_orders":    "⏳ Cargando pedidos pendientes",
    "get_packing_list":      "📋 Generando lista de empaquetado",
    "update_order_status":   "✏️ Actualizando estado del pedido",
    "list_order_statuses":   "📋 Consultando estados disponibles",
    "create_product":        "🆕 Creando producto en la BD",
}


def chat_with_image(
    image_b64: str,
    mime_type: str = "image/jpeg",
    message: str = "",
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Analiza una imagen de producto con Gemini y pasa el resultado al agente.

    Flujo:
      1. Llama a vision.analyze_product_image → JSON con atributos
      2. Inyecta el JSON como contexto en el mensaje del usuario
      3. Invoca el grafo del agente normalmente (agent ↔ tools loop)

    Returns:
        dict con reply, thread_id y vision_data (el JSON de Gemini).
    """
    from app.services.vision import analyze_product_image

    if not thread_id:
        thread_id = str(uuid.uuid4())

    alog.info("═" * 60)
    alog.info("[ADMIN IMAGE] thread=%s  mime=%s", thread_id[:8], mime_type)

    # ── 1. Análisis de imagen con Gemini Flash Lite ──────────────
    try:
        vision_data = analyze_product_image(image_b64, mime_type)
    except RuntimeError as exc:
        logger.error("Vision analysis failed: %s", exc)
        return {
            "reply": f"⚠️ No pude analizar la imagen: {exc}",
            "thread_id": thread_id,
            "vision_data": None,
        }

    alog.info("[ADMIN VISION] %s", _json.dumps(vision_data, ensure_ascii=False)[:300])

    # ── 2. Construir mensaje con el contexto de visión ───────────
    vision_json = _json.dumps(vision_data, ensure_ascii=False, indent=2)

    user_text = (
        f"[VISION_ANALYSIS]\n"
        f"He subido una foto de un producto. El modelo de visión ha extraído "
        f"estos atributos:\n\n```json\n{vision_json}\n```\n\n"
    )
    if message.strip():
        user_text += f"Mensaje adicional del admin: {message.strip()}\n\n"

    user_text += (
        "Por favor:\n"
        "1. Muéstrame un resumen de lo detectado.\n"
        "2. Pregúntame el precio y cualquier detalle que quiera cambiar.\n"
        "3. Cuando yo confirme, usa create_product para insertarlo."
    )

    # ── 3. Invocar el grafo ──────────────────────────────────────
    graph = _get_graph()
    input_state: AdminState = {
        "messages": [HumanMessage(content=user_text)],
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(input_state, config=config)

        reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        if not reply:
            reply = "Disculpa, no pude procesar el análisis de la imagen."

        alog.info("[ADMIN IMAGE REPLY] %s", reply[:200].replace("\n", " "))
        alog.info("═" * 60)

        return {
            "reply": reply,
            "thread_id": thread_id,
            "vision_data": vision_data,
        }
    except Exception as exc:
        logger.error("Error en admin chat_with_image: %s", exc, exc_info=True)
        return {
            "reply": (
                f"⚠️ Error técnico al procesar la imagen:\n\n"
                f"```\n{type(exc).__name__}: {exc}\n```"
            ),
            "thread_id": thread_id,
            "vision_data": vision_data,
        }


def chat_stream(
    message: str,
    thread_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Versión streaming del agente admin (SSE).

    Eventos:
      {"type": "status", "content": "..."}
      {"type": "result", "data": {"reply": ..., "thread_id": ...}}
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    graph = _get_graph()
    input_state: AdminState = {"messages": [HumanMessage(content=message)]}
    config = {"configurable": {"thread_id": thread_id}}

    alog.info("═" * 60)
    alog.info("[ADMIN STREAM] thread=%s", thread_id[:8])
    alog.info("[ADMIN USER] %r", message)

    yield {"type": "status", "content": "🧠 Analizando consulta..."}

    try:
        for event in graph.stream(input_state, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                if node_name == "agent":
                    for msg in state_update.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                label = _TOOL_LABELS.get(
                                    tc["name"], f"🔧 Ejecutando {tc['name']}"
                                )
                                yield {"type": "status", "content": label}
                        elif isinstance(msg, AIMessage) and msg.content:
                            yield {"type": "status", "content": "✍️ Redactando respuesta..."}
                elif node_name == "tools":
                    yield {"type": "status", "content": "⚙️ Procesando datos..."}

        snapshot = graph.get_state(config)
        all_messages = snapshot.values.get("messages", [])

        reply = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        if not reply:
            reply = "Disculpa, no pude procesar tu consulta."

        alog.info("[ADMIN REPLY/STREAM] %s", reply[:200].replace("\n", " "))
        alog.info("═" * 60)

        yield {
            "type": "result",
            "data": {"reply": reply, "thread_id": thread_id},
        }

    except Exception as exc:
        logger.error("Error en admin chat_stream: %s", exc, exc_info=True)
        yield {
            "type": "result",
            "data": {
                "reply": (
                    f"⚠️ Error técnico en el agente ERP:\n\n"
                    f"```\n{type(exc).__name__}: {exc}\n```"
                ),
                "thread_id": thread_id,
            },
        }
