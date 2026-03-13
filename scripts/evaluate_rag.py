"""
scripts/evaluate_rag.py — Pipeline de Evaluación del Sistema RAG
=================================================================
Evalúa la calidad del sistema RAG del agente UniArt Minerales usando
un Golden Dataset curado manualmente.

Métricas calculadas:
  • Mineral Discovery Rate (MDR): % de minerales esperados encontrados
    por infer_minerals_for_intent.
  • Catalog Hit Rate (CHR): % de búsquedas que devuelven ≥1 producto.
  • Context Precision@K: % de resultados RAG top-K relevantes.
  • Latencia media por componente (ms).
  • Coste estimado de tokens (si se usa OpenAI/OpenRouter).

Uso:
    python -m scripts.evaluate_rag
    python -m scripts.evaluate_rag --output evaluation_results/rag_v2.json
    python -m scripts.evaluate_rag --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# ══════════════════════════════════════════════════════════════════
# GOLDEN DATASET
# Estructura: lista de casos de prueba, cada uno con:
#   - query: pregunta del cliente
#   - type: "conceptual" | "direct" | "kb"
#   - expected_minerals: minerales que DEBEN aparecer en los resultados
#     de infer_minerals_for_intent (para queries conceptuales)
#   - expected_keywords: palabras que deben aparecer en los productos
#     devueltos por search_catalog
#   - notes: descripción del caso de prueba
# ══════════════════════════════════════════════════════════════════

GOLDEN_DATASET = [
    # ── Queries conceptuales (infer_minerals_for_intent) ──────────
    {
        "query": "que tienes para el amor",
        "type": "conceptual",
        "expected_minerals": ["cuarzo rosa"],
        "expected_keywords": ["cuarzo rosa", "corazón", "amor"],
        "notes": "Caso crítico: cuarzo rosa es la piedad del amor por excelencia",
    },
    {
        "query": "piedras para proteger la casa",
        "type": "conceptual",
        "expected_minerals": ["turmalina", "obsidiana"],
        "expected_keywords": ["protección", "turmalina", "obsidiana"],
        "notes": "Protección del hogar — minerales clásicos",
    },
    {
        "query": "algo para dormir mejor y reducir el estrés",
        "type": "conceptual",
        "expected_minerals": ["amatista", "selenita"],
        "expected_keywords": ["amatista", "selenita", "tranquilidad"],
        "notes": "Minerales relajantes y para el sueño",
    },
    {
        "query": "limpiar las energías de la casa",
        "type": "conceptual",
        "expected_minerals": ["amatista", "turmalina"],
        "expected_keywords": ["amatista", "selenita", "energía"],
        "notes": "Limpieza energética — selenita y turmalina negra son clave",
    },
    {
        "query": "minerales para meditar",
        "type": "conceptual",
        "expected_minerals": ["amatista"],
        "expected_keywords": ["amatista", "meditación", "cuarzo"],
        "notes": "Meditación — amatista es la más asociada",
    },
    {
        "query": "algo para la suerte y la prosperidad",
        "type": "conceptual",
        "expected_minerals": ["aventurina", "citrino"],
        "expected_keywords": ["aventurina", "citrino", "abundancia"],
        "notes": "Prosperidad — aventurina verde y citrino son clásicos",
    },
    # ── Queries directas de producto ─────────────────────────────
    {
        "query": "corazones de cuarzo rosa",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["cuarzo rosa", "corazón"],
        "notes": "Búsqueda directa de producto específico — debe encontrar resultados",
    },
    {
        "query": "amatista drusa",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["amatista"],
        "notes": "Drusa de amatista — producto concreto",
    },
    {
        "query": "colgante turmalina negra",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["turmalina"],
        "notes": "Colgante protección — búsqueda directa",
    },
    {
        "query": "pulseras de piedras naturales",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["pulsera"],
        "notes": "Categoría de producto — debe devolver variedad",
    },
    # ── Queries de base de conocimiento ──────────────────────────
    {
        "query": "cómo se limpia la amatista",
        "type": "kb",
        "expected_minerals": [],
        "expected_keywords": ["amatista", "limpi"],
        "notes": "Pregunta informativa sobre cuidado de minerales",
    },
    {
        "query": "política de devoluciones",
        "type": "kb",
        "expected_minerals": [],
        "expected_keywords": ["devolución", "devolver"],
        "notes": "Política de la tienda — debe encontrar en CMS",
    },
    # ── Edge cases ────────────────────────────────────────────────
    {
        "query": "ojo turco",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["ojo turco", "nazar", "ojo"],
        "notes": "Amuleto específico — puede llamarse nazar o azabache",
    },
    {
        "query": "orgonita",
        "type": "direct",
        "expected_minerals": [],
        "expected_keywords": ["orgonita"],
        "notes": "Búsqueda de orgonita específica",
    },
    {
        "query": "regalo para una persona muy espiritual",
        "type": "conceptual",
        "expected_minerals": ["amatista", "cuarzo"],
        "expected_keywords": ["amatista", "cuarzo", "selenita"],
        "notes": "Query vaga — debe extraer minerales espirituales del contexto",
    },
]


# ══════════════════════════════════════════════════════════════════
# EVALUADORES POR COMPONENTE
# ══════════════════════════════════════════════════════════════════

def evaluate_mineral_discovery(case: dict, verbose: bool = False) -> dict:
    """
    Evalúa infer_minerals_for_intent: ¿Descubre los minerales esperados?
    Solo aplica a casos tipo 'conceptual'.
    """
    if case["type"] != "conceptual":
        return {"skipped": True}

    from backend.agent import infer_minerals_for_intent

    t0 = time.perf_counter()
    result = infer_minerals_for_intent.invoke({"intent": case["query"]})
    latency_ms = (time.perf_counter() - t0) * 1000

    if result == "MINERALS_NOT_FOUND":
        found_minerals = []
    else:
        # Extraer minerales del resultado (líneas con "  - mineral")
        found_minerals = []
        for line in result.splitlines():
            if line.strip().startswith("- "):
                found_minerals.append(line.strip()[2:].lower())

    expected = [m.lower() for m in case["expected_minerals"]]
    hits = [m for m in expected if any(m in found for found in found_minerals)]
    mdr = len(hits) / len(expected) if expected else 1.0

    if verbose:
        print(f"\n  [MDR] query={case['query']!r}")
        print(f"        expected={expected}")
        print(f"        found={found_minerals}")
        print(f"        hits={hits}  MDR={mdr:.0%}  latency={latency_ms:.0f}ms")

    return {
        "mdr": mdr,
        "found_minerals": found_minerals,
        "expected_minerals": expected,
        "hits": hits,
        "latency_ms": round(latency_ms, 1),
        "raw_output_len": len(result),
    }


def evaluate_catalog_search(case: dict, verbose: bool = False) -> dict:
    """
    Evalúa search_catalog: ¿Devuelve productos relevantes?
    """
    from backend.agent import search_catalog

    query = case["query"]
    t0 = time.perf_counter()
    result = search_catalog.invoke({"query": query, "limit": 10})
    latency_ms = (time.perf_counter() - t0) * 1000

    found_products = [
        line for line in result.splitlines()
        if line.startswith("• **")
    ]
    num_products = len(found_products)
    hit = num_products > 0

    # Context Precision: % de productos que contienen alguna keyword esperada
    expected_kw = [kw.lower() for kw in case.get("expected_keywords", [])]
    precision_hits = 0
    for prod_line in found_products:
        prod_lower = prod_line.lower()
        if any(kw in prod_lower for kw in expected_kw):
            precision_hits += 1

    context_precision = precision_hits / num_products if num_products > 0 else 0.0

    if verbose:
        print(f"\n  [CHR] query={query!r}")
        print(f"        productos={num_products}  hit={hit}")
        print(f"        precision@{num_products}={context_precision:.0%}  latency={latency_ms:.0f}ms")
        for p in found_products[:3]:
            print(f"        {p[:80]}")

    return {
        "hit": hit,
        "num_products": num_products,
        "context_precision": round(context_precision, 3),
        "latency_ms": round(latency_ms, 1),
    }


def evaluate_kb_search(case: dict, verbose: bool = False) -> dict:
    """
    Evalúa search_knowledge_base: ¿Devuelve contenido relevante?
    Solo aplica a casos tipo 'kb'.
    """
    if case["type"] != "kb":
        return {"skipped": True}

    from backend.agent import search_knowledge_base

    t0 = time.perf_counter()
    result = search_knowledge_base.invoke({"question": case["query"]})
    latency_ms = (time.perf_counter() - t0) * 1000

    found_content = result != "No encontré información específica sobre esa consulta en la base de conocimiento."
    expected_kw = [kw.lower() for kw in case.get("expected_keywords", [])]
    result_lower = result.lower()
    kw_hits = [kw for kw in expected_kw if kw in result_lower]
    kw_precision = len(kw_hits) / len(expected_kw) if expected_kw else 1.0

    if verbose:
        print(f"\n  [KB]  query={case['query']!r}")
        print(f"        found_content={found_content}  kw_hits={kw_hits}")
        print(f"        kw_precision={kw_precision:.0%}  latency={latency_ms:.0f}ms")

    return {
        "found_content": found_content,
        "kw_precision": round(kw_precision, 3),
        "kw_hits": kw_hits,
        "latency_ms": round(latency_ms, 1),
        "result_len": len(result),
    }


# ══════════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def run_evaluation(verbose: bool = False) -> dict:
    print("\n" + "=" * 60)
    print("  🔬 Evaluación RAG — UniArt Minerales")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset: {len(GOLDEN_DATASET)} casos")
    print("=" * 60)

    results = []

    for i, case in enumerate(GOLDEN_DATASET, 1):
        print(f"\n[{i:02d}/{len(GOLDEN_DATASET)}] {case['type'].upper():12s} | {case['query'][:60]}")

        case_result = {
            "query": case["query"],
            "type": case["type"],
            "notes": case["notes"],
        }

        # 1. Mineral Discovery (solo conceptuales)
        mdr = evaluate_mineral_discovery(case, verbose=verbose)
        if not mdr.get("skipped"):
            case_result["mineral_discovery"] = mdr
            score = mdr["mdr"]
            print(f"   MDR={score:.0%}  minerals_found={mdr['found_minerals']}")

        # 2. Catalog search
        chr_result = evaluate_catalog_search(case, verbose=verbose)
        case_result["catalog_search"] = chr_result
        print(f"   CHR={'✅' if chr_result['hit'] else '❌'}  "
              f"n_products={chr_result['num_products']}  "
              f"precision={chr_result['context_precision']:.0%}  "
              f"latency={chr_result['latency_ms']:.0f}ms")

        # 3. KB search (solo tipo kb)
        kb = evaluate_kb_search(case, verbose=verbose)
        if not kb.get("skipped"):
            case_result["kb_search"] = kb
            print(f"   KB={'✅' if kb['found_content'] else '❌'}  "
                  f"kw_precision={kb['kw_precision']:.0%}  "
                  f"latency={kb['latency_ms']:.0f}ms")

        results.append(case_result)

    # ── Resumen ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 RESUMEN")
    print("=" * 60)

    # Catalog Hit Rate global
    catalog_hits = [r["catalog_search"]["hit"] for r in results]
    chr_global = sum(catalog_hits) / len(catalog_hits)

    # Context Precision global
    precisions = [r["catalog_search"]["context_precision"] for r in results
                  if r["catalog_search"]["num_products"] > 0]
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    # MDR global (solo conceptuales)
    mdr_scores = [r["mineral_discovery"]["mdr"]
                  for r in results if "mineral_discovery" in r]
    avg_mdr = sum(mdr_scores) / len(mdr_scores) if mdr_scores else None

    # Latencias
    catalog_latencies = [r["catalog_search"]["latency_ms"] for r in results]
    avg_latency = sum(catalog_latencies) / len(catalog_latencies)

    print(f"\n  Catalog Hit Rate (CHR):    {chr_global:.1%}  ({sum(catalog_hits)}/{len(catalog_hits)} queries con productos)")
    print(f"  Context Precision@K:       {avg_precision:.1%}  (relevancia de resultados)")
    if avg_mdr is not None:
        print(f"  Mineral Discovery Rate:    {avg_mdr:.1%}  ({len(mdr_scores)} queries conceptuales)")
    print(f"  Latencia media catálogo:   {avg_latency:.0f}ms")
    print()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(GOLDEN_DATASET),
        "metrics": {
            "catalog_hit_rate": round(chr_global, 4),
            "context_precision_avg": round(avg_precision, 4),
            "mineral_discovery_rate": round(avg_mdr, 4) if avg_mdr is not None else None,
            "avg_catalog_latency_ms": round(avg_latency, 1),
        },
        "cases": results,
    }

    return summary


# ══════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación RAG del agente UniArt")
    parser.add_argument(
        "--output", "-o",
        default=f"evaluation_results/rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Archivo de salida JSON",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar detalles")
    args = parser.parse_args()

    summary = run_evaluation(verbose=args.verbose)

    # Guardar resultados
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Resultados guardados en: {out_path}")
    print()
