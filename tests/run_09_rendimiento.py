"""
tests/run_09_rendimiento.py — Tests de rendimiento
Ejecutar con: python tests/run_09_rendimiento.py
"""
import json, sys, time, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_BASE = "http://localhost:8000"
ENDPOINTS = [
    ("GET", "/api/products", None),
    ("GET", "/api/customers/top", None),
    ("GET", "/api/admin/stats", None),
    ("GET", "/api/admin/model-status", None),
    ("GET", "/api/recommendations/100", None),
    ("GET", "/api/recommendations/999999", None),  # fallback
]

def measure_endpoint(method, path, payload, n=20):
    """Mide latencia de un endpoint N veces."""
    latencies = []
    errors = 0
    for _ in range(n):
        t0 = time.time()
        try:
            if method == "GET":
                r = requests.get(f"{API_BASE}{path}", timeout=30)
            else:
                r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
            ms = (time.time() - t0) * 1000
            if r.status_code == 200:
                latencies.append(ms)
            else:
                errors += 1
        except:
            errors += 1
    return latencies, errors

def percentile(data, p):
    if not data: return 0
    k = (len(data)-1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    d = k - f
    data_s = sorted(data)
    return data_s[f] + d * (data_s[c] - data_s[f])

def main():
    print("="*60+"\n  TEST 09: Rendimiento y Latencia\n"+"="*60)
    try: requests.get(f"{API_BASE}/docs", timeout=5)
    except: print(f" No se puede conectar a {API_BASE}"); sys.exit(1)

    N_REQUESTS = 20
    results = {}

    # ── Latencia individual por endpoint ──
    print(f"\n─── Latencia por endpoint ({N_REQUESTS} peticiones cada uno) ───\n")
    print(f"  {'Endpoint':<40} {'p50':>8} {'p95':>8} {'p99':>8} {'avg':>8} {'err':>5}")
    print(f"  {'─'*73}")

    for method, path, payload in ENDPOINTS:
        latencies, errors = measure_endpoint(method, path, payload, N_REQUESTS)
        if latencies:
            p50 = percentile(latencies, 50)
            p95 = percentile(latencies, 95)
            p99 = percentile(latencies, 99)
            avg = statistics.mean(latencies)
            print(f"  {method} {path:<36} {p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms {avg:>7.1f}ms {errors:>4}")
            results[path] = {"p50": round(p50,1), "p95": round(p95,1),
                             "p99": round(p99,1), "avg": round(avg,1),
                             "errors": errors, "n": N_REQUESTS}
        else:
            print(f"  {method} {path:<36} {'ERROR':>8} {'':>8} {'':>8} {'':>8} {errors:>4}")
            results[path] = {"error": True, "errors": errors}

    # ── Throughput: peticiones concurrentes ──
    print(f"\n─── Throughput (peticiones concurrentes a /api/products) ───\n")
    for n_workers in [5, 10, 20, 50]:
        latencies = []
        errors = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(requests.get, f"{API_BASE}/api/products", timeout=30)
                       for _ in range(n_workers)]
            for f in as_completed(futures):
                try:
                    r = f.result()
                    if r.status_code == 200:
                        latencies.append(r.elapsed.total_seconds() * 1000)
                    else:
                        errors += 1
                except:
                    errors += 1
        total_time = time.time() - t0
        rps = n_workers / total_time if total_time > 0 else 0
        avg_lat = statistics.mean(latencies) if latencies else 0
        print(f"  {n_workers:>3} concurrentes: {rps:>6.1f} req/s, "
              f"avg={avg_lat:>7.1f}ms, errors={errors}")
        results[f"concurrent_{n_workers}"] = {
            "workers": n_workers, "rps": round(rps,1),
            "avg_ms": round(avg_lat,1), "errors": errors}

    # ── Latencia de recomendaciones (inferencia ML) ──
    print(f"\n─── Latencia de inferencia ML (recomendaciones) ───\n")
    # Top customers para variar user_ids
    custs = requests.get(f"{API_BASE}/api/customers/top", timeout=10).json()
    ml_latencies = []
    for c in custs:
        t0 = time.time()
        r = requests.get(f"{API_BASE}/api/recommendations/{c['id_customer']}", timeout=30)
        ms = (time.time() - t0) * 1000
        if r.status_code == 200:
            ml_latencies.append(ms)

    if ml_latencies:
        print(f"  {len(ml_latencies)} inferencias ML:")
        print(f"    p50 = {percentile(ml_latencies, 50):.1f}ms")
        print(f"    p95 = {percentile(ml_latencies, 95):.1f}ms")
        print(f"    avg = {statistics.mean(ml_latencies):.1f}ms")
        print(f"    max = {max(ml_latencies):.1f}ms")
        results["ml_inference"] = {
            "n": len(ml_latencies),
            "p50": round(percentile(ml_latencies, 50), 1),
            "p95": round(percentile(ml_latencies, 95), 1),
            "avg": round(statistics.mean(ml_latencies), 1),
            "max": round(max(ml_latencies), 1),
        }

    # ── Resumen ──
    print(f"\n{'='*60}")
    print(f"  RESUMEN DE RENDIMIENTO")
    print(f"{'='*60}")
    for path, data in results.items():
        if isinstance(data, dict) and "p50" in data:
            status = "✅" if data["p50"] < 500 else "⚠"
            print(f"  {status} {path}: p50={data['p50']}ms, p95={data['p95']}ms")

    out = Path(__file__).resolve().parent / "results_09_rendimiento.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n Resultados: {out}")

if __name__ == "__main__":
    main()