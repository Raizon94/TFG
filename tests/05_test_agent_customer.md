# 05 — Tests del Agente de Cliente

> **Objetivo**: Evaluar el agente de cliente end-to-end, verificando que invoca las herramientas correctas (RAG, catálogo, KB, recomendaciones, pedidos, datos de cliente) y devuelve respuestas coherentes.
>
> **Requisito**: Backend corriendo, OPENAI_API_KEY configurada, ChromaDB indexado

---

## Herramientas evaluadas

| Herramienta | Caso de test | Descripción |
|---|---|---|
| `infer_minerals_for_intent` | RAG: intención → minerales | Descubrimiento de minerales por intención del usuario |
| `search_catalog` | Búsqueda directa | Búsqueda de productos en el catálogo |
| `browse_categories` | Exploración de categorías | Lista de categorías disponibles |
| `search_knowledge_base` | Consultas KB | Políticas de envío, devoluciones |
| `get_recommendations` | Recomendaciones ML | Recomendaciones personalizadas por historial |
| `get_order_status` | Estado de pedidos | Consulta de pedidos del cliente |
| `get_customer_info` | Datos del cliente | Información personal registrada |

---

## Metodología

1. Se envía cada query al endpoint **streaming** (`POST /api/chat/customer/stream`)
2. Se capturan los **status messages** del SSE para detectar qué herramientas invoca el agente
3. Se verifica que la respuesta contiene las **keywords esperadas** (mínimo configurable por caso)
4. Fallback al endpoint síncrono si el stream falla
5. Se registran latencia, keywords encontradas y herramientas detectadas

---

## Cómo ejecutar

```bash
cd /Users/bilian/Desktop/TFG
python tests/run_05_agent_customer.py
```

---

## Resultados

Guardados en `tests/results_05_agent_customer.json` con formato:

```json
{
  "total": 8,
  "passed": N,
  "failed": N,
  "infra_errors": N,
  "avg_latency_ms": N,
  "details": [...],
  "metrics": {
    "rag_minerals": {"tests_passed": N, "tests_total": N, ...},
    "catalog_search": {...},
    ...
  }
}
```
