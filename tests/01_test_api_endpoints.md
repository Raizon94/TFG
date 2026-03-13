# 01 — Tests de Endpoints API REST

> **Objetivo**: Verificar que todos los endpoints de la API FastAPI responden correctamente, con schemas válidos y tiempos de respuesta aceptables.
>
> **Requisito**: Backend corriendo en `http://localhost:8000`

---

## Script de prueba

```python
"""
tests/run_01_api.py — Tests de endpoints API REST
===================================================
Ejecutar con: python tests/run_01_api.py

Requisito: backend corriendo en http://localhost:8000
"""

import json
import time
import sys
import requests

API_BASE = "http://localhost:8000"

# ─── Utilidades ───────────────────────────────────────────────────

class TestResult:
    def __init__(self):
        self.results = []

    def add(self, name, passed, detail="", latency_ms=0):
        self.results.append({
            "name": name,
            "passed": passed,
            "detail": detail,
            "latency_ms": round(latency_ms, 2),
        })
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name} ({latency_ms:.0f}ms) — {detail}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        avg_latency = sum(r["latency_ms"] for r in self.results) / total if total else 0

        print(f"\n{'=' * 60}")
        print(f"  RESUMEN: {passed}/{total} tests pasados, {failed} fallidos")
        print(f"  Latencia media: {avg_latency:.1f} ms")
        print(f"{'=' * 60}")

        # Tabla de latencias
        print(f"\n  {'Endpoint':<45} {'Latencia (ms)':>12} {'Estado':>8}")
        print(f"  {'─' * 67}")
        for r in self.results:
            icon = "✅" if r["passed"] else "❌"
            print(f"  {r['name']:<45} {r['latency_ms']:>10.1f}ms {icon:>8}")

        return {"total": total, "passed": passed, "failed": failed,
                "avg_latency_ms": round(avg_latency, 2), "details": self.results}


def timed_get(url, **kwargs):
    t0 = time.time()
    r = requests.get(url, timeout=30, **kwargs)
    return r, (time.time() - t0) * 1000


def timed_post(url, **kwargs):
    t0 = time.time()
    r = requests.post(url, timeout=30, **kwargs)
    return r, (time.time() - t0) * 1000


# ─── Tests ────────────────────────────────────────────────────────

def test_health(tr: TestResult):
    """Verificar que la API está viva (Swagger docs)."""
    try:
        r, ms = timed_get(f"{API_BASE}/docs")
        tr.add("GET /docs (health)", r.status_code == 200,
               f"status={r.status_code}", ms)
    except Exception as e:
        tr.add("GET /docs (health)", False, str(e))


def test_products(tr: TestResult):
    """GET /api/products — Debe devolver lista de productos con estructura correcta."""
    r, ms = timed_get(f"{API_BASE}/api/products")
    passed = r.status_code == 200
    data = r.json() if passed else []

    # Validar estructura
    if passed and len(data) > 0:
        p = data[0]
        has_fields = all(k in p for k in ["id", "name", "price", "image_url"])
        passed = passed and has_fields and len(data) >= 4
        detail = f"{len(data)} productos, campos={'OK' if has_fields else 'FALTAN'}"
    else:
        detail = f"status={r.status_code}, len={len(data)}"

    tr.add("GET /api/products", passed, detail, ms)


def test_recommendations_known_user(tr: TestResult):
    """GET /api/recommendations/{user_id} — Usuario con historial."""
    # Usamos un usuario que sabemos que tiene compras (top customer)
    r_cust, _ = timed_get(f"{API_BASE}/api/customers/top")
    if r_cust.status_code == 200 and r_cust.json():
        user_id = r_cust.json()[0]["id_customer"]
    else:
        user_id = 1

    r, ms = timed_get(f"{API_BASE}/api/recommendations/{user_id}")
    passed = r.status_code == 200
    data = r.json() if passed else {}

    if passed:
        recs = data.get("recommendations", [])
        passed = len(recs) == 3
        ids = [p["id"] for p in recs]
        detail = f"{len(recs)} recomendaciones, IDs={ids}"
    else:
        detail = f"status={r.status_code}"

    tr.add(f"GET /api/recommendations/{user_id} (conocido)", passed, detail, ms)


def test_recommendations_unknown_user(tr: TestResult):
    """GET /api/recommendations/{user_id} — Usuario sin historial → fallback."""
    r, ms = timed_get(f"{API_BASE}/api/recommendations/999999")
    passed = r.status_code == 200
    data = r.json() if passed else {}

    if passed:
        recs = data.get("recommendations", [])
        passed = len(recs) == 3  # Fallback debe devolver 3
        detail = f"{len(recs)} recs (fallback), IDs={[p['id'] for p in recs]}"
    else:
        detail = f"status={r.status_code}"

    tr.add("GET /api/recommendations/999999 (fallback)", passed, detail, ms)


def test_top_customers(tr: TestResult):
    """GET /api/customers/top — Top 10 clientes por gasto."""
    r, ms = timed_get(f"{API_BASE}/api/customers/top")
    passed = r.status_code == 200
    data = r.json() if passed else []

    if passed and len(data) > 0:
        c = data[0]
        has_fields = all(k in c for k in ["id_customer", "name", "email", "total_spent"])
        sorted_ok = all(data[i]["total_spent"] >= data[i+1]["total_spent"]
                        for i in range(len(data)-1))
        passed = has_fields and len(data) == 10 and sorted_ok
        detail = f"{len(data)} clientes, ordenados={'OK' if sorted_ok else 'NO'}"
    else:
        detail = f"status={r.status_code}, len={len(data)}"

    tr.add("GET /api/customers/top", passed, detail, ms)


def test_admin_stats(tr: TestResult):
    """GET /api/admin/stats — KPIs del dashboard."""
    r, ms = timed_get(f"{API_BASE}/api/admin/stats")
    passed = r.status_code == 200
    data = r.json() if passed else {}

    if passed:
        fields = ["total_pedidos", "ingresos_mes", "clientes_activos",
                  "productos_activos", "tasa_conversion", "ticket_medio"]
        has_fields = all(k in data for k in fields)
        passed = has_fields and data["total_pedidos"] > 0
        detail = (f"pedidos={data.get('total_pedidos')}, "
                  f"ticket_medio={data.get('ticket_medio')}")
    else:
        detail = f"status={r.status_code}"

    tr.add("GET /api/admin/stats", passed, detail, ms)


def test_model_status(tr: TestResult):
    """GET /api/admin/model-status — Estado del modelo ML."""
    r, ms = timed_get(f"{API_BASE}/api/admin/model-status")
    passed = r.status_code == 200
    data = r.json() if passed else {}

    if passed:
        passed = data.get("svd_loaded") is True
        detail = (f"svd_loaded={data.get('svd_loaded')}, "
                  f"size={data.get('svd_file_size_kb')}KB, "
                  f"retrain_needed={data.get('should_retrain')}")
    else:
        detail = f"status={r.status_code}"

    tr.add("GET /api/admin/model-status", passed, detail, ms)


def test_checkout_empty_cart(tr: TestResult):
    """POST /api/checkout — Carrito vacío debe retornar 400."""
    payload = {"id_customer": 1, "id_address": 1, "items": []}
    r, ms = timed_post(f"{API_BASE}/api/checkout", json=payload)
    passed = r.status_code == 400
    tr.add("POST /api/checkout (vacío → 400)", passed,
           f"status={r.status_code}", ms)


def test_checkout_invalid_customer(tr: TestResult):
    """POST /api/checkout — Cliente inexistente debe retornar 404."""
    payload = {
        "id_customer": 999999,
        "id_address": 1,
        "items": [{"product_id": 1, "product_name": "Test", "unit_price": 10.0, "quantity": 1}]
    }
    r, ms = timed_post(f"{API_BASE}/api/checkout", json=payload)
    passed = r.status_code == 404
    tr.add("POST /api/checkout (cliente inválido → 404)", passed,
           f"status={r.status_code}", ms)


def test_boost_list(tr: TestResult):
    """GET /api/admin/boost — Lista de boosts activos."""
    r, ms = timed_get(f"{API_BASE}/api/admin/boost")
    passed = r.status_code == 200
    data = r.json() if passed else None
    tr.add("GET /api/admin/boost", passed,
           f"status={r.status_code}, count={len(data) if isinstance(data, list) else 'N/A'}", ms)


def test_boost_search(tr: TestResult):
    """GET /api/admin/boost/search?q=amatista — Búsqueda de productos para boost."""
    r, ms = timed_get(f"{API_BASE}/api/admin/boost/search", params={"q": "amatista"})
    passed = r.status_code == 200
    data = r.json() if passed else []
    tr.add("GET /api/admin/boost/search?q=amatista", passed,
           f"status={r.status_code}, resultados={len(data)}", ms)


def test_retrain_log(tr: TestResult):
    """GET /api/admin/retrain-log — Log del último reentrenamiento."""
    r, ms = timed_get(f"{API_BASE}/api/admin/retrain-log")
    passed = r.status_code == 200
    data = r.json() if passed else {}
    log_preview = (data.get("log", "")[:80] + "...") if data.get("log") else "vacío"
    tr.add("GET /api/admin/retrain-log", passed,
           f"log={log_preview}", ms)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TEST 01: Endpoints API REST")
    print("=" * 60)

    # Verificar que la API está corriendo
    try:
        requests.get(f"{API_BASE}/docs", timeout=5)
    except Exception:
        print(f"\n  ❌ ERROR: No se puede conectar a {API_BASE}")
        print("  Asegúrate de que el backend está corriendo:")
        print("    uvicorn backend.app.main:app --port 8000")
        sys.exit(1)

    tr = TestResult()

    print("\n─── Salud ───")
    test_health(tr)

    print("\n─── Catálogo ───")
    test_products(tr)

    print("\n─── Recomendaciones ───")
    test_recommendations_known_user(tr)
    test_recommendations_unknown_user(tr)

    print("\n─── Clientes ───")
    test_top_customers(tr)

    print("\n─── Admin ───")
    test_admin_stats(tr)
    test_model_status(tr)
    test_retrain_log(tr)

    print("\n─── Boost ───")
    test_boost_list(tr)
    test_boost_search(tr)

    print("\n─── Checkout (errores esperados) ───")
    test_checkout_empty_cart(tr)
    test_checkout_invalid_customer(tr)

    summary = tr.summary()

    # Guardar resultados JSON
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parent / "results_01_api.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Resultados guardados: {out}")


if __name__ == "__main__":
    main()
```

---

## Métricas esperadas

| Métrica | Criterio de éxito |
|---|---|
| Tests pasados | 12/12 (100%) |
| Latencia media | < 500ms |
| Latencia p95 | < 1000ms |
| Schemas válidos | Todos con campos requeridos |
| Códigos de error | 400 y 404 donde corresponde |

---

## Cómo ejecutar

```bash
cd /Users/bilian/Desktop/TFG
python tests/run_01_api.py
```
