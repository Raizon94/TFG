"""
tests/run_06_agent_admin.py — Conversaciones con el Agente de Administración
==============================================================================
Ejecutar con: python tests/run_06_agent_admin.py

Requisito: Backend corriendo. OPENAI_API_KEY configurada.
Envía consultas al agente de administración ERP que ejercitan cada
herramienta y registra pregunta, respuesta y latencia.
"""

import json
import sys
import time
import uuid
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"

# ─── Conversaciones ──────────────────────────────────────────────
# Cada entrada fuerza al agente admin a usar una herramienta concreta.

CONVERSATIONS = [
    {
        "message": "Dame un informe de ventas de este mes",
        "tool_target": "get_sales_report",
        "description": "Informe de ventas del período",
    },
    {
        "message": "¿Cuáles son los 5 productos más vendidos?",
        "tool_target": "get_top_products",
        "description": "Ranking de productos más vendidos",
    },
    {
        "message": "Busca clientes que se llamen María",
        "tool_target": "search_customers",
        "description": "Búsqueda de clientes por nombre",
    },
    {
        "message": "Dame las estadísticas generales del catálogo de productos",
        "tool_target": "get_catalog_stats",
        "description": "Estadísticas del catálogo",
    },
    {
        "message": "¿Hay pedidos pendientes de enviar?",
        "tool_target": "get_pending_orders",
        "description": "Pedidos pendientes de envío",
    },
    {
        "message": "¿Qué estados de pedido existen en el sistema?",
        "tool_target": "list_order_statuses",
        "description": "Lista de estados de pedido",
    },
    {
        "message": "Busca los últimos pedidos realizados",
        "tool_target": "search_orders",
        "description": "Búsqueda de pedidos recientes",
    },
    {
        "message": "Dame los detalles completos del pedido número 8",
        "tool_target": "get_order_details",
        "description": "Detalles de un pedido específico",
    },
]


def query_admin(message):
    """Envía una query al agente admin via endpoint síncrono."""
    tid = str(uuid.uuid4())
    resp = requests.post(
        f"{API_BASE}/api/chat/admin",
        json={"message": message, "thread_id": tid},
        timeout=120,
    )
    if resp.status_code == 200:
        data = resp.json()
        return data.get("reply", "")
    return ""


def main():
    print("=" * 60)
    print("  TEST 06: Agente de Administración (conversaciones)")
    print("=" * 60)

    # Verificar backend disponible
    try:
        requests.get(f"{API_BASE}/docs", timeout=5)
    except Exception:
        print(f"\n No se puede conectar a {API_BASE}")
        sys.exit(1)

    results = []

    for i, conv in enumerate(CONVERSATIONS, 1):
        print(f"\n─── [{i}/{len(CONVERSATIONS)}] {conv['description']} ───")
        print(f" Pregunta: {conv['message']}")

        t0 = time.time()
        try:
            reply = query_admin(conv["message"])
            ms = (time.time() - t0) * 1000

            print(f" Respuesta: {reply[:200]}{'…' if len(reply) > 200 else ''}")
            print(f"  {ms:.0f}ms")

            results.append({
                "description": conv["description"],
                "tool_target": conv["tool_target"],
                "question": conv["message"],
                "answer": reply,
                "latency_ms": round(ms, 2),
            })

        except Exception as e:
            ms = (time.time() - t0) * 1000
            print(f" Error: {e}")
            results.append({
                "description": conv["description"],
                "tool_target": conv["tool_target"],
                "question": conv["message"],
                "answer": f"ERROR: {e}",
                "latency_ms": round(ms, 2),
            })

    # ── Resumen ──────────────────────────────────────────────────
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    print(f"\n{'=' * 60}")
    print(f"  {len(results)} conversaciones completadas")
    print(f"  Latencia media: {avg_latency:.0f}ms")
    print(f"{'=' * 60}")

    output = {
        "total": len(results),
        "avg_latency_ms": round(avg_latency, 2),
        "conversations": results,
    }

    out = Path(__file__).resolve().parent / "results_06_agent_admin.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n Resultados guardados: {out}")


if __name__ == "__main__":
    main()
