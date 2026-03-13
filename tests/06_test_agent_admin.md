# 06 — Tests del Agente de Administración

> **Objetivo**: Evaluar el agente de administración ERP end-to-end, verificando que invoca las herramientas correctas (ventas, pedidos, clientes, catálogo) y devuelve datos coherentes de gestión.
>
> **Requisito**: Backend corriendo, OPENAI_API_KEY configurada

---

## Herramientas evaluadas

| Herramienta | Caso de test | Descripción |
|---|---|---|
| `get_sales_report` | Informe de ventas | KPIs: pedidos, ingresos, ticket medio |
| `get_top_products` | Top productos | Ranking por unidades vendidas |
| `search_customers` | Buscar clientes | Búsqueda por nombre/email |
| `get_catalog_stats` | Estadísticas catálogo | Productos activos, precios, categorías |
| `get_pending_orders` | Pedidos pendientes | Pedidos con pago aceptado sin enviar |
| `list_order_statuses` | Estados de pedido | Lista de estados disponibles |
| `search_orders` | Buscar pedidos | Búsqueda por filtros |
| `get_order_details` | Detalle de pedido | Líneas, cliente, historial de un pedido |

---

## Metodología

1. Se envía cada query al endpoint síncrono (`POST /api/chat/admin`)
2. Se verifica que la respuesta contiene las **keywords esperadas** que demuestran que la herramienta fue invocada y devolvió datos reales de la BD
3. Se registran latencia y keywords encontradas por categoría

---

## Cómo ejecutar

```bash
cd /Users/bilian/Desktop/TFG
python tests/run_06_agent_admin.py
```

---

## Resultados

Guardados en `tests/results_06_agent_admin.json` con formato:

```json
{
  "total": 8,
  "passed": N,
  "failed": N,
  "infra_errors": N,
  "avg_latency_ms": N,
  "details": [...],
  "metrics": {
    "sales_report": {"tests_passed": N, "tests_total": N, ...},
    "top_products": {...},
    ...
  }
}
```
