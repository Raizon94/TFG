# 04 — Tests del Sistema RAG

> **Objetivo**: Evaluar la calidad del pipeline RAG (Retrieval-Augmented Generation) del agente de ventas: descubrimiento de minerales, búsqueda en catálogo, y base de conocimiento.
>
> **Requisito**: Backend corriendo, ChromaDB con datos RAG indexados

---

## Script de prueba

```python
"""
tests/run_04_rag.py — Tests del sistema RAG
=============================================
Ejecutar con: python tests/run_04_rag.py

Requisito: Backend corriendo. ChromaDB indexado.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Golden Dataset ───────────────────────────────────────────────

GOLDEN_DATASET = [
    # Conceptuales
    {
        "query": "que tienes para el amor",
        "type": "conceptual",
        "expected_minerals": ["cuarzo rosa"],
        "expected_keywords": ["cuarzo rosa", "amor", "corazón"],
        "notes": "Amor y relaciones — cuarzo rosa es el mineral clave",
    },
    {
        "query": "necesito protección energética",
        "type": "conceptual",
        "expected_minerals": ["turmalina"],
        "expected_keywords": ["turmalina", "protección", "negra"],
        "notes": "Protección — turmalina negra es la referencia",
    },
    {
        "query": "limpieza energética del hogar",
        "type": "conceptual",
        "expected_minerals": ["amatista", "turmalina"],
        "expected_keywords": ["amatista", "selenita", "energía"],
        "notes": "Limpieza energética — selenita y turmalina negra",
    },
    {
        "query": "minerales para meditar",
        "type": "conceptual",
        "expected_minerals": ["amatista"],
        "expected_keywords": ["amatista", "meditación", "cuarzo"],
        "notes": "Meditación — amatista es la más asociada",
    },
    {
        "query": "regalo para una persona muy espiritual",
        "type": "conceptual",
        "expected_minerals": ["amatista", "cuarzo"],
        "expected_keywords": ["amatista", "cuarzo", "selenita"],
        "notes": "Query vaga — debe extraer minerales espirituales",
    },
    # Directas
    {
        "query": "colgante turmalina negra",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["turmalina"],
        "notes": "Búsqueda directa de producto",
    },
    {
        "query": "pulseras de piedras naturales",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["pulsera"],
        "notes": "Categoría — debe devolver variedad",
    },
    {
        "query": "geodas de amatista",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["geoda", "amatista"],
        "notes": "Producto específico",
    },
    {
        "query": "orgonita",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["orgonita"],
        "notes": "Búsqueda de orgonita",
    },
    # Knowledge Base
    {
        "query": "¿cuánto tarda el envío?",
        "type": "kb",
        "expected_minerals": [],
        "expected_keywords": ["envío", "plazo", "entrega"],
        "notes": "Política de envíos",
    },
    {
        "query": "¿puedo devolver un producto?",
        "type": "kb",
        "expected_minerals": [],
        "expected_keywords": ["devolución", "devolver", "plazo"],
        "notes": "Política de devoluciones",
    },
    {
        "query": "¿qué métodos de pago aceptáis?",
        "type": "kb",
        "expected_minerals": [],
        "expected_keywords": ["pago", "tarjeta", "transferencia"],
        "notes": "Métodos de pago",
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
        print(f"\n{'=' * 60}")
        print(f"  RESUMEN: {passed}/{total} tests pasados ({passed/total*100:.0f}%)")
        print(f"{'=' * 60}")
        return {"total": total, "passed": passed, "details": self.results}


def test_mineral_discovery(tr: TestResult):
    """Evalúa infer_minerals_for_intent para queries conceptuales."""
    from app.agents.customer.tools import infer_minerals_for_intent

    conceptual_cases = [c for c in GOLDEN_DATASET if c["type"] == "conceptual"]
    total_expected = 0
    total_found = 0

    print(f"\n  Evaluando {len(conceptual_cases)} queries conceptuales...\n")

    for case in conceptual_cases:
        t0 = time.time()
        try:
            result = infer_minerals_for_intent(case["query"])
            ms = (time.time() - t0) * 1000
            result_lower = result.lower() if isinstance(result, str) else str(result).lower()

            found = []
            for mineral in case["expected_minerals"]:
                total_expected += 1
                if mineral.lower() in result_lower:
                    found.append(mineral)
                    total_found += 1

            passed = len(found) == len(case["expected_minerals"])
            tr.add(
                f"Mineral Discovery: '{case['query']}'",
                passed,
                f"esperados={case['expected_minerals']}, encontrados={found}",
                ms
            )
        except Exception as e:
            ms = (time.time() - t0) * 1000
            tr.add(f"Mineral Discovery: '{case['query']}'", False, f"Error: {e}", ms)

    recall = total_found / total_expected * 100 if total_expected else 0
    print(f"\n  📊 Mineral Discovery Recall: {total_found}/{total_expected} ({recall:.0f}%)")
    return recall


def test_catalog_search(tr: TestResult):
    """Evalúa search_catalog para queries directas y conceptuales."""
    from app.agents.customer.tools import search_catalog

    all_cases = [c for c in GOLDEN_DATASET if c["type"] in ("direct", "conceptual")]
    total_kw = 0
    found_kw = 0

    print(f"\n  Evaluando {len(all_cases)} búsquedas en catálogo...\n")

    for case in all_cases:
        t0 = time.time()
        try:
            results = search_catalog(case["query"])
            ms = (time.time() - t0) * 1000

            results_str = str(results).lower()
            found = []
            for kw in case["expected_keywords"]:
                total_kw += 1
                if kw.lower() in results_str:
                    found.append(kw)
                    found_kw += 1

            n_results = len(results) if isinstance(results, list) else 0
            passed = len(found) > 0
            tr.add(
                f"Catalog Search: '{case['query']}'",
                passed,
                f"{n_results} resultados, keywords={found}/{case['expected_keywords']}",
                ms
            )
        except Exception as e:
            ms = (time.time() - t0) * 1000
            tr.add(f"Catalog Search: '{case['query']}'", False, f"Error: {e}", ms)

    precision = found_kw / total_kw * 100 if total_kw else 0
    print(f"\n  📊 Catalog Keyword Precision: {found_kw}/{total_kw} ({precision:.0f}%)")
    return precision


def test_knowledge_base(tr: TestResult):
    """Evalúa search_knowledge_base para queries de KB."""
    from app.agents.customer.tools import search_knowledge_base

    kb_cases = [c for c in GOLDEN_DATASET if c["type"] == "kb"]
    total_kw = 0
    found_kw = 0

    print(f"\n  Evaluando {len(kb_cases)} queries de Knowledge Base...\n")

    for case in kb_cases:
        t0 = time.time()
        try:
            results = search_knowledge_base(case["query"])
            ms = (time.time() - t0) * 1000

            results_str = str(results).lower()
            found = []
            for kw in case["expected_keywords"]:
                total_kw += 1
                if kw.lower() in results_str:
                    found.append(kw)
                    found_kw += 1

            passed = len(found) > 0
            tr.add(
                f"KB Search: '{case['query']}'",
                passed,
                f"keywords={found}/{case['expected_keywords']}",
                ms
            )
        except Exception as e:
            ms = (time.time() - t0) * 1000
            tr.add(f"KB Search: '{case['query']}'", False, f"Error: {e}", ms)

    recall = found_kw / total_kw * 100 if total_kw else 0
    print(f"\n  📊 KB Keyword Recall: {found_kw}/{total_kw} ({recall:.0f}%)")
    return recall


def main():
    print("=" * 60)
    print("  TEST 04: Sistema RAG (Retrieval-Augmented Generation)")
    print("=" * 60)

    tr = TestResult()

    print("\n─── Mineral Discovery (queries conceptuales → minerales) ───")
    mineral_recall = test_mineral_discovery(tr)

    print("\n─── Catalog Search (queries → productos relevantes) ───")
    catalog_precision = test_catalog_search(tr)

    print("\n─── Knowledge Base (políticas, envíos, pagos) ───")
    kb_recall = test_knowledge_base(tr)

    # Resumen global de métricas
    summary = tr.summary()
    summary["metrics"] = {
        "mineral_discovery_recall": round(mineral_recall, 1),
        "catalog_keyword_precision": round(catalog_precision, 1),
        "kb_keyword_recall": round(kb_recall, 1),
    }

    print(f"\n  📊 Métricas globales RAG:")
    print(f"     Mineral Discovery Recall:  {mineral_recall:.1f}%")
    print(f"     Catalog Keyword Precision: {catalog_precision:.1f}%")
    print(f"     KB Keyword Recall:         {kb_recall:.1f}%")

    out = Path(__file__).resolve().parent / "results_04_rag.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Resultados guardados: {out}")


if __name__ == "__main__":
    main()
```

---

## Métricas esperadas

| Componente | Métrica | Criterio |
|---|---|---|
| Mineral Discovery | Recall | > 70% |
| Catalog Search | Keyword Precision | > 50% |
| Knowledge Base | Keyword Recall | > 60% |

---

## Cómo ejecutar

```bash
cd /Users/bilian/Desktop/TFG
python tests/run_04_rag.py
```
