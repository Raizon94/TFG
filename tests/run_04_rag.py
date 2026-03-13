"""
tests/run_04_rag.py — Tests del sistema RAG via API
=====================================================
Ejecutar con: python tests/run_04_rag.py

Requisito: Backend corriendo. OPENAI_API_KEY configurada.
Evalúa el RAG a través del agente (endpoint /api/chat/customer)
enviando queries conceptuales and verificando que las respuestas
contienen los minerales/keywords esperados.
"""

import json
import sys
import time
import uuid
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"
THINKING_STATUS = "🧠 Pensando..."
NEUTRAL_STATUSES = {
    THINKING_STATUS,
    "⚙️ Procesando resultados...",
    "✍️ Redactando respuesta...",
}
ERROR_MARKERS = (
    "⚠️ error técnico en el agente",
    "connection error",
    "apiconnectionerror",
    "network is unreachable",
    "revisa la configuración del llm",
)

# ─── Golden Dataset ───────────────────────────────────────────────

GOLDEN_DATASET = [
    # Conceptuales (RAG → mineral discovery)
    {
        "query": "que tienes para el amor",
        "type": "conceptual",
        "expected_keywords": ["cuarzo rosa", "amor", "corazón"],
        "notes": "Amor y relaciones — cuarzo rosa",
    },
    {
        "query": "necesito protección energética",
        "type": "conceptual",
        "expected_keywords": ["turmalina", "protección", "negra", "obsidiana"],
        "notes": "Protección — turmalina negra",
    },
    {
        "query": "limpieza energética del hogar",
        "type": "conceptual",
        "expected_keywords": ["selenita", "amatista", "energía", "limpieza", "turmalina"],
        "notes": "Limpieza — selenita, amatista",
    },
    {
        "query": "minerales para meditar",
        "type": "conceptual",
        "expected_keywords": ["amatista", "meditación", "cuarzo", "calma"],
        "notes": "Meditación — amatista",
    },
    # Directas (catalog search)
    {
        "query": "quiero ver pulseras",
        "type": "direct",
        "expected_keywords": ["pulsera"],
        "notes": "Categoría de producto",
    },
    {
        "query": "tienes orgonitas?",
        "type": "direct",
        "expected_keywords": ["orgonita"],
        "notes": "Producto específico",
    },
    # Knowledge Base
    {
        "query": "¿cuánto tarda el envío?",
        "type": "kb",
        "expected_keywords": ["envío", "plazo", "entrega", "día"],
        "notes": "Política de envíos",
    },
    {
        "query": "¿puedo devolver un producto?",
        "type": "kb",
        "expected_keywords": ["devolución", "devolver", "plazo", "cambio"],
        "notes": "Política de devoluciones",
    },
]


class TestResult:
    def __init__(self):
        self.results = []

    def add(self, name, passed, detail="", latency_ms=0):
        self.results.append({"name": name, "passed": passed,
                             "detail": detail, "latency_ms": round(latency_ms, 2)})
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name} ({latency_ms:.0f}ms)")
        if detail:
            print(f"      {detail}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        infra_errors = sum(1 for r in self.results if r.get("infra_error"))
        avg_latency = sum(r["latency_ms"] for r in self.results) / total if total else 0
        print(f"\n{'=' * 60}")
        print(f"  RESUMEN: {passed}/{total} tests pasados ({passed/total*100:.0f}%)")
        print(f"  Latencia media: {avg_latency:.0f}ms")
        if infra_errors:
            print(f"  Incidencias de infraestructura: {infra_errors}")
        print(f"{'=' * 60}")
        return {"total": total, "passed": passed,
                "infra_errors": infra_errors,
                "avg_latency_ms": round(avg_latency, 2), "details": self.results}


def query_agent(message, id_customer=100):
    """Envía una query al agente via endpoint síncrono."""
    tid = str(uuid.uuid4())
    resp = requests.post(
        f"{API_BASE}/api/chat/customer",
        json={"message": message, "id_customer": id_customer, "thread_id": tid},
        timeout=120
    )
    if resp.status_code == 200:
        data = resp.json()
        return data.get("reply", ""), data.get("products", [])
    return "", []


def query_agent_stream(message, id_customer=100):
    """Envía query via SSE stream para detectar tools y respuesta."""
    tid = str(uuid.uuid4())
    resp = requests.post(
        f"{API_BASE}/api/chat/customer/stream",
        json={"message": message, "id_customer": id_customer, "thread_id": tid},
        stream=True, timeout=120
    )
    reply = ""
    tools = []
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            try:
                evt = json.loads(line[6:])
                if evt.get("type") == "result":
                    reply = evt.get("data", {}).get("reply", "")
                elif evt.get("type") == "status":
                    tools.append(evt.get("content", ""))
            except json.JSONDecodeError:
                pass
    return reply, tools


def _is_error_reply(reply: str) -> bool:
    low = (reply or "").lower()
    return any(marker in low for marker in ERROR_MARKERS)


def _filter_meaningful_statuses(statuses: list[str]) -> list[str]:
    """Elimina estados neutros o repetidos del stream."""
    filtered: list[str] = []
    seen: set[str] = set()
    for status in statuses:
        clean = (status or "").strip()
        if not clean or clean in NEUTRAL_STATUSES or clean in seen:
            continue
        seen.add(clean)
        filtered.append(clean)
    return filtered


def _looks_like_infra_failure(reply: str) -> bool:
    """Solo marca infra si hay error explícito o respuesta vacía."""
    return _is_error_reply(reply) or not (reply or "").strip()


def main():
    print("=" * 60)
    print("  TEST 04: Sistema RAG (via API del Agente)")
    print("=" * 60)

    try:
        requests.get(f"{API_BASE}/docs", timeout=5)
    except Exception:
        print(f"\n  ❌ No se puede conectar a {API_BASE}")
        sys.exit(1)

    tr = TestResult()
    infra_failures = 0

    # Métricas por tipo
    metrics = {"conceptual": {"total_kw": 0, "found_kw": 0},
               "direct":     {"total_kw": 0, "found_kw": 0},
               "kb":         {"total_kw": 0, "found_kw": 0}}

    for case in GOLDEN_DATASET:
        qtype = case["type"]
        label = {"conceptual": "🔮 Conceptual", "direct": "🔍 Catálogo", "kb": "📚 KB"}[qtype]
        print(f"\n─── {label}: \"{case['query']}\" ───")

        t0 = time.time()
        try:
            reply, tools = query_agent_stream(case["query"])
            if not reply or _is_error_reply(reply):
                reply, _ = query_agent(case["query"])
            ms = (time.time() - t0) * 1000

            meaningful_tools = _filter_meaningful_statuses(tools)
            infra_error = _looks_like_infra_failure(reply)
            if infra_error:
                infra_failures += 1

            reply_lower = reply.lower()
            found = []
            for kw in case["expected_keywords"]:
                metrics[qtype]["total_kw"] += 1
                if kw.lower() in reply_lower:
                    found.append(kw)
                    metrics[qtype]["found_kw"] += 1

            passed = len(found) > 0 and not infra_error
            if meaningful_tools:
                tools_str = ", ".join(meaningful_tools)
            elif tools:
                tools_str = "solo estado inicial/neutral"
            else:
                tools_str = "ninguna"
            detail = (
                f"keywords encontradas: {found}/{case['expected_keywords']} | "
                f"tools: {tools_str}"
            )
            if infra_error:
                detail += f" | posible fallo de infraestructura/LLM: {reply[:140]}"

            tr.results.append({
                "name": f"{qtype}: '{case['query']}'",
                "passed": passed,
                "detail": detail,
                "latency_ms": round(ms, 2),
                "infra_error": infra_error,
            })
            icon = "✅" if passed else "❌"
            print(f"  {icon} {qtype}: '{case['query']}' ({ms:.0f}ms)")
            print(f"      {detail}")
        except Exception as e:
            ms = (time.time() - t0) * 1000
            infra_failures += 1
            tr.results.append({
                "name": f"{qtype}: '{case['query']}'",
                "passed": False,
                "detail": f"Error: {e}",
                "latency_ms": round(ms, 2),
                "infra_error": True,
            })
            print(f"  ❌ {qtype}: '{case['query']}' ({ms:.0f}ms)")
            print(f"      Error: {e}")

    # Resumen
    summary = tr.summary()

    # Métricas por tipo
    print(f"\n  📊 Métricas RAG por tipo:")
    for qtype, m in metrics.items():
        pct = m["found_kw"] / m["total_kw"] * 100 if m["total_kw"] else 0
        print(f"     {qtype:>12}: {m['found_kw']}/{m['total_kw']} keywords ({pct:.0f}%)")

    summary["metrics"] = {
        k: {"found": v["found_kw"], "total": v["total_kw"],
            "pct": round(v["found_kw"] / v["total_kw"] * 100, 1) if v["total_kw"] else 0}
        for k, v in metrics.items()
    }

    out = Path(__file__).resolve().parent / "results_04_rag.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  📄 Resultados guardados: {out}")

    total = summary["total"]
    passed = summary["passed"]
    if infra_failures == total:
        print("\n  ❌ Fallo global de infraestructura/LLM: el agente no pudo ejecutar el flujo RAG.")
        sys.exit(2)
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
