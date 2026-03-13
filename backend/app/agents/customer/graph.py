"""
backend/graph.py — Grafo LangGraph del agente de cliente
=========================================================
Define AgentState, los nodos (guardrail, agent, tools, check_escalation),
los routers y compila el grafo con checkpointer persistente.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.config import alog, get_llm, logger, memory
from app.agents.customer.prompts import SYSTEM_PROMPT
from app.agents.customer.tools import TOOLS


# ═══════════════════════════════════════════════════════════════════
# Estado del grafo
# ═══════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Estado del grafo conversacional."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    id_customer: int | None
    escalated: bool


# ═══════════════════════════════════════════════════════════════════
# Constructor del grafo
# ═══════════════════════════════════════════════════════════════════

def _build_graph() -> StateGraph:
    """Construye y compila el grafo LangGraph del agente cliente."""

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # ── Nodo: Agente (razona y decide si usar tools) ──────────────
    def agent_node(state: AgentState) -> dict:
        """Nodo principal: el LLM razona con el contexto y decide."""
        messages = list(state["messages"])

        # Inyectar contexto del cliente si lo tenemos
        id_customer = state.get("id_customer")
        system_msg = SYSTEM_PROMPT
        if id_customer:
            system_msg += (
                f"\n\n**Contexto de sesión:** El cliente actual tiene "
                f"ID #{id_customer}. Pasa SIEMPRE id_customer={id_customer} "
                f"a search_catalog, get_recommendations, get_order_status "
                f"y get_customer_info. No se lo pidas al cliente."
            )

        # Asegurar que el system prompt está al inicio
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_msg))
        else:
            messages[0] = SystemMessage(content=system_msg)

        try:
            response = llm_with_tools.invoke(messages)
            # ── Log decisión del LLM ──
            if isinstance(response, AIMessage) and response.tool_calls:
                calls_desc = ", ".join(
                    f"{tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc.get('args', {}).items())})"
                    for tc in response.tool_calls
                )
                alog.info(
                    "  [LLM] → TOOL CALLS (%d): %s",
                    len(response.tool_calls),
                    calls_desc,
                )
            elif isinstance(response, AIMessage) and response.content:
                alog.info(
                    "  [LLM] → RESPUESTA FINAL (%d chars)", len(response.content)
                )
        except Exception as e:
            err_str = str(e).lower()

            # Rate-limit — wait and retry once
            if "rate" in err_str or "too many" in err_str or "429" in err_str:
                logger.warning("Rate-limited — waiting 3 s before retry…")
                time.sleep(3)
                try:
                    response = llm_with_tools.invoke(messages)
                except Exception as retry_exc:
                    response = AIMessage(
                        content=(
                            "El servicio de IA está temporalmente saturado. "
                            "Por favor, espera unos segundos e inténtalo "
                            "de nuevo.\n\n"
                            f"```\n{type(retry_exc).__name__}: {retry_exc}\n```"
                        )
                    )

            # Azure/GitHub Models content filter false positive
            elif "content_filter" in err_str:
                logger.warning(
                    "Content filter triggered — retrying without tools: %s",
                    str(e)[:120],
                )
                try:
                    response = llm.invoke(messages)
                except Exception:
                    logger.warning(
                        "Content filter also blocked retry. "
                        "Returning local fallback response."
                    )
                    response = AIMessage(
                        content=(
                            "Disculpa, no he podido procesar tu consulta "
                            "en este momento debido a un filtro externo del "
                            "proveedor de IA. Puedes intentar reformular "
                            "tu pregunta o contactar directamente con nuestro "
                            "equipo. Tambien puedo ayudarte con otras "
                            "consultas sobre la tienda, pedidos o "
                            "recomendaciones."
                        )
                    )
            else:
                raise

        return {"messages": [response]}

    # ── Nodo: Ejecutar herramientas ───────────────────────────────
    tool_node = ToolNode(TOOLS)

    # ── Nodo: Post-procesado de escalado ──────────────────────────
    def check_escalation(state: AgentState) -> dict:
        """Verifica si la última tool fue un escalado."""
        messages = state["messages"]
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if "ESCALADO_CONFIRMADO" in msg.content:
                    return {"escalated": True}
                break
        return {"escalated": False}

    # ── Router: decide siguiente nodo ─────────────────────────────
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Decide si el agente necesita ejecutar tools o ya terminó."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    # ── Nodo: Guardrail de entrada ────────────────────────────────
    def input_guardrail_node(state: AgentState) -> dict:
        """Rechaza mensajes maliciosos o fuera de ámbito antes del agente."""
        from app.agents.shared.guardrails import check_input as _check_input

        last = state["messages"][-1] if state["messages"] else None
        msg_text = last.content if last and hasattr(last, "content") else ""
        is_safe, rejection = _check_input(str(msg_text))
        if not is_safe:
            alog.info("  [GUARDRAIL] BLOQUEADO: %.80s", msg_text)
            return {"messages": [AIMessage(content=rejection)]}
        return {}

    def route_after_guardrail(state: AgentState) -> Literal["agent", "end"]:
        """Si el guardrail añadió un rechazo, termina; si no, va al agente."""
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage):
            return "end"
        return "agent"

    # ── Construir grafo ───────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("check_escalation", check_escalation)

    graph.set_entry_point("input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"agent": "agent", "end": END},
    )

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # Después de tools → check escalation → volver al agente
    graph.add_edge("tools", "check_escalation")
    graph.add_edge("check_escalation", "agent")

    return graph.compile(checkpointer=memory)


# ═══════════════════════════════════════════════════════════════════
# Singleton del grafo compilado
# ═══════════════════════════════════════════════════════════════════

_compiled_graph = None


def get_graph():
    """Obtiene o crea el grafo compilado (lazy init)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph
