# 03 — Evaluación Temporal SVD: Entrena en X, Verifica en X+1

> **Objetivo**: Evaluar la capacidad predictiva del modelo SVD usando **corte temporal real**: entrena con datos históricos hasta un punto X y verifica si las recomendaciones aciertan con las compras **reales** del período X+1.
>
> **Requisito**: MySQL corriendo con `tfg_bd`

---

## Metodología

```
──────────────────────────────────────────────────────────────►  tiempo
│◄─── TRAIN (hasta X%) ───►│◄── TEST (X% → 100%) ──►│
│                            │                          │
│  Entrena SVD aquí          │  Verifica aquí           │
│  Calcula popular ranking   │  ¿Acertó las compras?   │
```

**Diferencia con `evaluate_ml.py`**: este script ejecuta **múltiples puntos de corte** (60%, 70%, 80%, 90%) y calcula métricas extendidas incluyendo **MAP, MRR, Coverage, y factor de mejora vs baseline**.

---

## Script de prueba

```python
"""
tests/run_03_temporal_svd.py — Evaluación Temporal Multi-Corte SVD
===================================================================
Ejecutar con: python tests/run_03_temporal_svd.py

Entrena SVD hasta un punto temporal X y verifica predicciones
contra compras reales en X+1. Ejecuta múltiples cutoffs.

Requisito: MySQL con tfg_bd
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sqlalchemy import create_engine

warnings.filterwarnings("ignore", category=FutureWarning)

DATABASE_URL = "mysql+pymysql://root:root1234@localhost:3306/tfg_bd"
RANDOM_STATE = 42
SVD_N_COMPONENTS = 50
K_VALUES = [3, 5, 10]
CUTOFF_POINTS = [0.60, 0.70, 0.80, 0.90]


# ═══════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════

def ndcg_at_k(recommended: list[int], actual: set[int], k: int) -> float:
    dcg = 0.0
    for i, pid in enumerate(recommended[:k]):
        if pid in actual:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: list[int], actual: set[int]) -> float:
    """AP: media de precision@i solo en posiciones con acierto."""
    hits = 0
    sum_prec = 0.0
    for i, pid in enumerate(recommended):
        if pid in actual:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(actual) if actual else 0.0


def reciprocal_rank(recommended: list[int], actual: set[int]) -> float:
    """RR: 1/posición del primer acierto."""
    for i, pid in enumerate(recommended):
        if pid in actual:
            return 1.0 / (i + 1)
    return 0.0


# ═══════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════

def load_data(engine):
    orders_df = pd.read_sql("""
        SELECT o.id_order, o.id_customer, o.date_add, o.total_paid_real
        FROM ps_orders o WHERE o.valid = 1 ORDER BY o.date_add
    """, engine, parse_dates=["date_add"])

    details_df = pd.read_sql("""
        SELECT od.id_order, od.product_id, od.product_quantity
        FROM ps_order_detail od
        JOIN ps_orders o ON od.id_order = o.id_order
        WHERE o.valid = 1
    """, engine)

    return orders_df, details_df


# ═══════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════

def build_svd(orders_train, details_train):
    merged = orders_train[["id_order", "id_customer"]].merge(
        details_train[["id_order", "product_id", "product_quantity"]],
        on="id_order",
    )
    interaction = (
        merged.groupby(["id_customer", "product_id"], as_index=False)
        .agg(qty=("product_quantity", "sum"))
    )

    customer_ids = sorted(interaction["id_customer"].unique())
    product_ids = sorted(interaction["product_id"].unique())
    cust_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
    prod_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    idx_to_prod = {i: pid for pid, i in prod_to_idx.items()}

    rows = interaction["id_customer"].map(cust_to_idx).values
    cols = interaction["product_id"].map(prod_to_idx).values
    vals = np.log1p(interaction["qty"].values.astype(float))

    matrix = csr_matrix((vals, (rows, cols)),
                        shape=(len(customer_ids), len(product_ids)))

    n_comp = min(SVD_N_COMPONENTS, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    svd.fit(matrix)

    return svd, matrix, cust_to_idx, prod_to_idx, idx_to_prod, product_ids


# ═══════════════════════════════════════════════════════════════════
# EVALUACIÓN SVD EN UN CUTOFF
# ═══════════════════════════════════════════════════════════════════

def evaluate_svd_at_cutoff(orders_train, details_train, orders_test, details_test):
    """Evalúa SVD: entrena con datos ≤ X, predice, verifica contra X+1."""
    svd, matrix, cust_to_idx, prod_to_idx, idx_to_prod, product_ids = (
        build_svd(orders_train, details_train)
    )

    # Ground truth: compras reales después del corte
    merged_test = orders_test[["id_order", "id_customer"]].merge(
        details_test[["id_order", "product_id"]], on="id_order")
    ground_truth = merged_test.groupby("id_customer")["product_id"].apply(set).to_dict()

    eval_customers = [c for c in ground_truth if c in cust_to_idx]
    max_k = max(K_VALUES)

    results_by_k = {k: {"hits": 0, "prec_sum": 0, "recall_sum": 0,
                         "ndcg_sum": 0} for k in K_VALUES}
    ap_sum = 0.0
    rr_sum = 0.0
    n_eval = 0
    all_recommended_products = set()

    for cust_id in eval_customers:
        actual = ground_truth[cust_id].intersection(prod_to_idx.keys())
        if not actual:
            continue

        user_idx = cust_to_idx[cust_id]
        user_vec = matrix[user_idx].toarray().reshape(1, -1)
        if user_vec.sum() == 0:
            continue

        user_latent = svd.transform(user_vec)
        scores = (user_latent @ svd.components_).flatten()
        purchased = matrix[user_idx].toarray().flatten()
        scores[purchased > 0] = -np.inf

        top_indices = np.argsort(scores)[::-1][:max_k]
        recommended = [idx_to_prod[int(i)] for i in top_indices]
        n_eval += 1

        all_recommended_products.update(recommended)
        ap_sum += average_precision(recommended, actual)
        rr_sum += reciprocal_rank(recommended, actual)

        for k in K_VALUES:
            top_k = recommended[:k]
            top_k_set = set(top_k)
            hits = len(top_k_set & actual)
            results_by_k[k]["hits"] += 1 if hits > 0 else 0
            results_by_k[k]["prec_sum"] += hits / k
            results_by_k[k]["recall_sum"] += hits / len(actual)
            results_by_k[k]["ndcg_sum"] += ndcg_at_k(top_k, actual, k)

    metrics = {
        "n_eval": n_eval,
        "variance_explained": float(svd.explained_variance_ratio_.sum()),
        "n_users_train": len(cust_to_idx),
        "n_products": len(product_ids),
        "MAP": ap_sum / n_eval if n_eval else 0,
        "MRR": rr_sum / n_eval if n_eval else 0,
        "coverage": len(all_recommended_products) / len(product_ids) * 100 if product_ids else 0,
    }

    for k in K_VALUES:
        r = results_by_k[k]
        metrics[f"hit_rate@{k}"] = r["hits"] / n_eval * 100 if n_eval else 0
        metrics[f"precision@{k}"] = r["prec_sum"] / n_eval * 100 if n_eval else 0
        metrics[f"recall@{k}"] = r["recall_sum"] / n_eval * 100 if n_eval else 0
        metrics[f"ndcg@{k}"] = r["ndcg_sum"] / n_eval if n_eval else 0

    return metrics


# ═══════════════════════════════════════════════════════════════════
# BASELINE POPULARIDAD
# ═══════════════════════════════════════════════════════════════════

def evaluate_popularity_at_cutoff(orders_train, details_train, orders_test, details_test):
    merged_train = orders_train[["id_order", "id_customer"]].merge(
        details_train[["id_order", "product_id", "product_quantity"]], on="id_order")
    popularity = merged_train.groupby("product_id")["product_quantity"].sum().sort_values(ascending=False)
    popular_ranking = list(popularity.index)
    customer_history = merged_train.groupby("id_customer")["product_id"].apply(set).to_dict()

    merged_test = orders_test[["id_order", "id_customer"]].merge(
        details_test[["id_order", "product_id"]], on="id_order")
    ground_truth = merged_test.groupby("id_customer")["product_id"].apply(set).to_dict()

    eval_customers = [c for c in ground_truth if c in customer_history]
    max_k = max(K_VALUES)

    results_by_k = {k: {"hits": 0, "prec_sum": 0, "recall_sum": 0,
                         "ndcg_sum": 0} for k in K_VALUES}
    ap_sum = 0.0
    rr_sum = 0.0
    n_eval = 0

    for cust_id in eval_customers:
        actual = ground_truth[cust_id]
        if not actual:
            continue
        already = customer_history.get(cust_id, set())
        recommended = [p for p in popular_ranking if p not in already][:max_k]
        if not recommended:
            continue
        n_eval += 1

        ap_sum += average_precision(recommended, actual)
        rr_sum += reciprocal_rank(recommended, actual)

        for k in K_VALUES:
            top_k = recommended[:k]
            hits = len(set(top_k) & actual)
            results_by_k[k]["hits"] += 1 if hits > 0 else 0
            results_by_k[k]["prec_sum"] += hits / k
            results_by_k[k]["recall_sum"] += hits / len(actual)
            results_by_k[k]["ndcg_sum"] += ndcg_at_k(top_k, actual, k)

    metrics = {"n_eval": n_eval, "MAP": ap_sum / n_eval if n_eval else 0,
               "MRR": rr_sum / n_eval if n_eval else 0}
    for k in K_VALUES:
        r = results_by_k[k]
        metrics[f"hit_rate@{k}"] = r["hits"] / n_eval * 100 if n_eval else 0
        metrics[f"precision@{k}"] = r["prec_sum"] / n_eval * 100 if n_eval else 0
        metrics[f"recall@{k}"] = r["recall_sum"] / n_eval * 100 if n_eval else 0
        metrics[f"ndcg@{k}"] = r["ndcg_sum"] / n_eval if n_eval else 0

    return metrics


# ═══════════════════════════════════════════════════════════════════
# MAIN: MULTI-CUTOFF EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  TEST 03: Evaluación Temporal SVD — Multi-Cutoff")
    print("  Entrena en tiempo X → Verifica compras reales en X+1")
    print("=" * 70)

    engine = create_engine(DATABASE_URL, echo=False)
    orders_df, details_df = load_data(engine)
    orders_sorted = orders_df.sort_values("date_add")

    all_results = {}

    for cutoff_pct in CUTOFF_POINTS:
        print(f"\n{'═' * 70}")
        print(f"  CUTOFF: {cutoff_pct:.0%} train / {1-cutoff_pct:.0%} test")
        print(f"{'═' * 70}")

        cutoff_idx = int(len(orders_sorted) * cutoff_pct)
        cutoff_date = orders_sorted.iloc[cutoff_idx]["date_add"]

        orders_train = orders_sorted[orders_sorted["date_add"] <= cutoff_date].copy()
        orders_test = orders_sorted[orders_sorted["date_add"] > cutoff_date].copy()

        train_ids = set(orders_train["id_order"])
        test_ids = set(orders_test["id_order"])
        details_train = details_df[details_df["id_order"].isin(train_ids)].copy()
        details_test = details_df[details_df["id_order"].isin(test_ids)].copy()

        date_range_train = f"{orders_train['date_add'].min().strftime('%Y-%m')} → {orders_train['date_add'].max().strftime('%Y-%m')}"
        date_range_test = f"{orders_test['date_add'].min().strftime('%Y-%m')} → {orders_test['date_add'].max().strftime('%Y-%m')}"

        # Reincidentes
        overlap = set(orders_train["id_customer"]) & set(orders_test["id_customer"])

        print(f"  Fecha corte: {cutoff_date.strftime('%Y-%m-%d')}")
        print(f"  TRAIN: {len(orders_train):,} pedidos ({date_range_train})")
        print(f"  TEST:  {len(orders_test):,} pedidos ({date_range_test})")
        print(f"  Clientes reincidentes: {len(overlap):,}")

        # ── SVD ──
        t0 = time.time()
        svd_metrics = evaluate_svd_at_cutoff(
            orders_train, details_train, orders_test, details_test)
        svd_time = time.time() - t0

        # ── Popularidad ──
        t0 = time.time()
        pop_metrics = evaluate_popularity_at_cutoff(
            orders_train, details_train, orders_test, details_test)
        pop_time = time.time() - t0

        # ── Mostrar resultados ──
        print(f"\n  {'Métrica':<22} {'SVD':>10} {'Popular':>10} {'Δ':>10} {'Factor':>10}")
        print(f"  {'─' * 62}")

        for k in K_VALUES:
            print(f"\n  --- K={k} ---")
            for metric, fmt in [("hit_rate", ".1f"), ("precision", ".1f"),
                                 ("recall", ".1f"), ("ndcg", ".3f")]:
                s = svd_metrics.get(f"{metric}@{k}", 0)
                p = pop_metrics.get(f"{metric}@{k}", 0)
                diff = s - p
                factor = s / p if p > 0 else float("inf")
                sign = "+" if diff >= 0 else ""
                suffix = "%" if metric != "ndcg" else ""
                print(f"  {metric}@{k:<5} {s:>9{fmt}}{suffix} {p:>9{fmt}}{suffix} "
                      f"{sign}{diff:>8{fmt}} {factor:>9.1f}×")

        # Métricas adicionales
        print(f"\n  --- Métricas Adicionales ---")
        print(f"  {'MAP':<22} {svd_metrics['MAP']:>10.4f} {pop_metrics['MAP']:>10.4f} "
              f"{svd_metrics['MAP']-pop_metrics['MAP']:>+10.4f}")
        print(f"  {'MRR':<22} {svd_metrics['MRR']:>10.4f} {pop_metrics['MRR']:>10.4f} "
              f"{svd_metrics['MRR']-pop_metrics['MRR']:>+10.4f}")
        print(f"  {'Cobertura (%)':<22} {svd_metrics.get('coverage', 0):>10.1f}")
        print(f"  {'Varianza expl.':<22} {svd_metrics.get('variance_explained', 0):>10.4f}")
        print(f"  {'Clientes evaluados':<22} {svd_metrics['n_eval']:>10}")
        print(f"  {'Tiempo SVD (s)':<22} {svd_time:>10.2f}")
        print(f"  {'Tiempo Popular (s)':<22} {pop_time:>10.2f}")

        all_results[f"cutoff_{cutoff_pct:.0%}"] = {
            "cutoff_pct": cutoff_pct,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "n_train": len(orders_train),
            "n_test": len(orders_test),
            "n_reincidentes": len(overlap),
            "svd": svd_metrics,
            "popularity": pop_metrics,
            "svd_time_s": round(svd_time, 2),
            "pop_time_s": round(pop_time, 2),
        }

    # ═══════════════════════════════════════════════════════════════
    # RESUMEN EJECUTIVO MULTI-CUTOFF
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("  📊 RESUMEN EJECUTIVO: SVD vs Popularidad en múltiples cortes")
    print(f"{'=' * 70}")

    print(f"\n  Hit Rate @3 por cutoff:")
    print(f"  {'Cutoff':<12} {'SVD':>10} {'Popular':>10} {'Factor×':>10}")
    print(f"  {'─' * 44}")
    for key, data in all_results.items():
        s = data["svd"].get("hit_rate@3", 0)
        p = data["popularity"].get("hit_rate@3", 0)
        f = s / p if p > 0 else float("inf")
        print(f"  {data['cutoff_pct']:.0%:<12} {s:>9.1f}% {p:>9.1f}% {f:>10.1f}×")

    print(f"\n  Hit Rate @10 por cutoff:")
    print(f"  {'Cutoff':<12} {'SVD':>10} {'Popular':>10} {'Factor×':>10}")
    print(f"  {'─' * 44}")
    for key, data in all_results.items():
        s = data["svd"].get("hit_rate@10", 0)
        p = data["popularity"].get("hit_rate@10", 0)
        f = s / p if p > 0 else float("inf")
        print(f"  {data['cutoff_pct']:.0%:<12} {s:>9.1f}% {p:>9.1f}% {f:>10.1f}×")

    # NDCG evolution
    print(f"\n  NDCG @5 por cutoff:")
    print(f"  {'Cutoff':<12} {'SVD':>10} {'Popular':>10} {'Factor×':>10}")
    print(f"  {'─' * 44}")
    for key, data in all_results.items():
        s = data["svd"].get("ndcg@5", 0)
        p = data["popularity"].get("ndcg@5", 0)
        f = s / p if p > 0 else float("inf")
        print(f"  {data['cutoff_pct']:.0%:<12} {s:>9.4f} {p:>9.4f} {f:>10.1f}×")

    # MAP y MRR evolution
    print(f"\n  MAP y MRR por cutoff:")
    print(f"  {'Cutoff':<12} {'MAP-SVD':>10} {'MAP-Pop':>10} {'MRR-SVD':>10} {'MRR-Pop':>10}")
    print(f"  {'─' * 56}")
    for key, data in all_results.items():
        print(f"  {data['cutoff_pct']:.0%:<12} "
              f"{data['svd']['MAP']:>10.4f} {data['popularity']['MAP']:>10.4f} "
              f"{data['svd']['MRR']:>10.4f} {data['popularity']['MRR']:>10.4f}")

    # Ganador global
    svd_wins = 0
    pop_wins = 0
    total_comparisons = 0
    for key, data in all_results.items():
        for k in K_VALUES:
            for metric in ["hit_rate", "precision", "recall", "ndcg"]:
                s = data["svd"].get(f"{metric}@{k}", 0)
                p = data["popularity"].get(f"{metric}@{k}", 0)
                total_comparisons += 1
                if s > p: svd_wins += 1
                elif p > s: pop_wins += 1

    print(f"\n  🏆 MARCADOR GLOBAL: SVD gana {svd_wins}/{total_comparisons}, "
          f"Popularidad {pop_wins}/{total_comparisons}")

    # Guardar resultados
    out = Path(__file__).resolve().parent / "results_03_temporal_svd.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  📄 Resultados guardados: {out}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
```

---

## Métricas por cutoff

| Cutoff | Métricas |
|---|---|
| 60% train / 40% test | Hit Rate, Precision, Recall, NDCG @{3,5,10} + MAP, MRR, Coverage |
| 70% train / 30% test | Ídem |
| 80% train / 20% test | Ídem (caso default, comparable con `evaluate_ml.py`) |
| 90% train / 10% test | Ídem |

### Métricas adicionales vs baseline

| Métrica | Descripción |
|---|---|
| **MAP** | Mean Average Precision — media de precision en cada posición con acierto |
| **MRR** | Mean Reciprocal Rank — 1/posición del primer acierto |
| **Coverage** | % del catálogo total que aparece en alguna recomendación |
| **Factor×** | Cuántas veces mejor es SVD vs popularidad |

---

## Cómo ejecutar

```bash
cd /Users/bilian/Desktop/TFG
python tests/run_03_temporal_svd.py
```
