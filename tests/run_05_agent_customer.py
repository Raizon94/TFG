"""
tests/run_05_agent_customer.py — Conversaciones con el Agente de Cliente
=========================================================================
Ejecutar con: python tests/run_05_agent_customer.py

Requisito: Backend corriendo. OPENAI_API_KEY configurada.
Envía consultas al agente de cliente que ejercitan cada herramienta
y registra pregunta, respuesta, herramientas invocadas y latencia.
"""

import json
import sys
import time
import uuid
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"
NEUTRAL_STATUSES = {
    "🧠 Pensando...",
    "⚙️ Procesando resultados...",
    "✍️ Redactando respuesta...",
}

# ─── Conversaciones ──────────────────────────────────────────────
# Cada entrada fuerza al agente a usar una o más herramientas.

CONVERSATIONS = [
    {
        "message": "Busco algo que me ayude con la ansiedad y el estrés",
        "id_customer": None,
        "tool_target": "infer_minerals_for_intent + search_catalog",
        "description": "RAG: descubrimiento de minerales para intención",
    },
    {
        "message": "Quiero ver colgantes de turmalina negra",
        "id_customer": None,
        "tool_target": "search_catalog",
        "description": "Búsqueda directa en el catálogo",
    },
    {
        "message": "¿Qué categorías de productos tenéis disponibles?",
        "id_customer": None,
        "tool_target": "browse_categories",
        "description": "Exploración de categorías del catálogo",
    },
    {
        "message": "¿Hacéis envíos a Canarias? ¿Cuánto tardan?",
        "id_customer": None,
        "tool_target": "search_knowledge_base",
        "description": "Consulta a la base de conocimiento (envíos)",
    },
    {
        "message": "¿Cuál es vuestra política de devoluciones?",
        "id_customer": None,
        "tool_target": "search_knowledge_base",
        "description": "Consulta a la base de conocimiento (devoluciones)",
    },
    {
        "message": "¿Qué me recomendáis basándoos en mis compras anteriores?",
        "id_customer": 100,
        "tool_target": "get_recommendations",
        "description": "Recomendaciones personalizadas (ML)",
    },
    {
        "message": "¿Cuál es el estado de mis pedidos?",
        "id_customer": 100,
        "tool_target": "get_order_status",
        "description": "Consulta de estado de pedidos del cliente",
    },
    {
        "message": "Quiero consultar mis datos de cliente registrados",
        "id_customer": 100,
        "tool_target": "get_customer_info",
        "description": "Consulta de información del cliente",
    },
]


def query_agent_stream(message, id_customer=None):
    """Envía query via SSE stream para capturar tools y respuesta."""
    tid = str(uuid.uuid4())
    payload = {"message": message, "thread_id": tid}
    if id_customer is not None:
        payload["id_customer"] = id_customer
    resp = requests.post(
        f"{API_BASE}/api/chat/customer/stream",
        json=payload, stream=True, timeout=120,
    )
    reply = ""
    statuses = []
    products = []
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            try:
                evt = json.loads(line[6:])
                if evt.get("type") == "result":
                    data = evt.get("data", {})
                    reply = data.get("reply", "")
                    products = data.get("products", [])
                elif evt.get("type") == "status":
                    statuses.append(evt.get("content", ""))
            except json.JSONDecodeError:
                pass
    return reply, statuses, products


def _filter_meaningful_statuses(statuses: list[str]) -> list[str]:
    seen: set[str] = set()
    filtered: list[str] = []
    for s in statuses:
        clean = (s or "").strip()
        if not clean or clean in NEUTRAL_STATUSES or clean in seen:
            continue
        seen.add(clean)
        filtered.append(clean)
    return filtered


def main():
    print("=" * 60)
    print("  TEST 05: Agente de Cliente (conversaciones)")
    print("=" * 60)

    # Verificar backend disponible
    try:
        requests.get(f"{API_BASE}/docs", timeout=5)
    except Exception:
        print(f"\n  ❌ No se puede conectar a {API_BASE}")
        sys.exit(1)

    results = []

    for i, conv in enumerate(CONVERSATIONS, 1):
        print(f"\n─── [{i}/{len(CONVERSATIONS)}] {conv['description']} ───")
        print(f"  📩 Pregunta: {conv['message']}")

        t0 = time.time()
        try:
            reply, statuses, products = query_agent_stream(
                conv["message"], conv["id_customer"],
            )
            ms = (time.time() - t0) * 1000
            tools = _filter_meaningful_statuses(statuses)

            print(f"  🤖 Respuesta: {reply[:200]}{'…' if len(reply) > 200 else ''}")
            if tools:
                print(f"  🔧 Tools: {', '.join(tools)}")
            if products:
                print(f"  📦 Productos devueltos: {len(products)}")
            print(f"  ⏱️  {ms:.0f}ms")

            results.append({
                "description": conv["description"],
                "tool_target": conv["tool_target"],
                "id_customer": conv["id_customer"],
                "question": conv["message"],
                "answer": reply,
                "tools_invoked": tools,
                "products_returned": len(products),
                "latency_ms": round(ms, 2),
            })

        except Exception as e:
            ms = (time.time() - t0) * 1000
            print(f"  ❌ Error: {e}")
            results.append({
                "description": conv["description"],
                "tool_target": conv["tool_target"],
                "id_customer": conv["id_customer"],
                "question": conv["message"],
                "answer": f"ERROR: {e}",
                "tools_invoked": [],
                "products_returned": 0,
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

    out = Path(__file__).resolve().parent / "results_05_agent_customer.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  📄 Resultados guardados: {out}")


if __name__ == "__main__":
    main()
