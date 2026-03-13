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
K_VALUES = [3, 5, 10]
CUTOFF_POINTS = [0.60, 0.70, 0.80, 0.90]
SVD_COMPONENTS_GRID = [20, 50, 100]


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

def build_svd(orders_train, details_train, n_components: int):
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

    n_comp = min(n_components, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    svd.fit(matrix)

    return svd, matrix, cust_to_idx, prod_to_idx, idx_to_prod, product_ids


# ═══════════════════════════════════════════════════════════════════
# EVALUACIÓN SVD EN UN CUTOFF
# ═══════════════════════════════════════════════════════════════════

def evaluate_svd_at_cutoff(
    orders_train,
    details_train,
    orders_test,
    details_test,
    n_components: int,
):
    """Evalúa SVD: entrena con datos ≤ X, predice, verifica contra X+1."""
    svd, matrix, cust_to_idx, prod_to_idx, idx_to_prod, product_ids = (
        build_svd(orders_train, details_train, n_components=n_components)
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
        "requested_components": int(n_components),
        "actual_components": int(svd.n_components),
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


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def _format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}"


def _format_factor(candidate: float, baseline: float) -> str:
    if baseline <= 0:
        return "inf"
    return f"{candidate / baseline:.1f}x"


def print_top5_comparison_table(pop_metrics: dict, svd_runs: list[dict]) -> None:
    """Imprime una tabla compacta centrada en top-5."""
    print("\n  Tabla principal Top-5 (baseline popularidad vs SVD)")
    print(
        f"  {'Comp':>5} {'Var':>7} {'Hit@5 Pop':>11} {'Hit@5 SVD':>11} "
        f"{'Δ Hit':>8} {'Factor':>8} {'Prec@5 SVD':>12} {'NDCG@5 SVD':>12}"
    )
    print(f"  {'─' * 88}")

    pop_hit5 = pop_metrics.get("hit_rate@5", 0.0)
    for run in svd_runs:
        m = run["metrics"]
        hit5 = m.get("hit_rate@5", 0.0)
        print(
            f"  {m['actual_components']:>5} "
            f"{m['variance_explained']:>6.3f} "
            f"{_format_pct(pop_hit5):>11} "
            f"{_format_pct(hit5):>11} "
            f"{_format_delta(hit5 - pop_hit5):>8} "
            f"{_format_factor(hit5, pop_hit5):>8} "
            f"{_format_pct(m.get('precision@5', 0.0)):>12} "
            f"{m.get('ndcg@5', 0.0):>12.4f}"
        )


def print_parameter_detail_table(pop_metrics: dict, svd_runs: list[dict]) -> None:
    """Imprime varias métricas @5 para cada configuración de SVD."""
    print("\n  Comparativa detallada @5 por parámetros SVD")
    print(
        f"  {'Comp':>5} {'Hit@5':>9} {'ΔHit':>8} {'Prec@5':>9} {'ΔPrec':>8} "
        f"{'Recall@5':>11} {'ΔRec':>8} {'NDCG@5':>9} {'ΔNDCG':>8}"
    )
    print(f"  {'─' * 90}")

    for run in svd_runs:
        m = run["metrics"]
        print(
            f"  {m['actual_components']:>5} "
            f"{_format_pct(m.get('hit_rate@5', 0.0)):>9} "
            f"{_format_delta(m.get('hit_rate@5', 0.0) - pop_metrics.get('hit_rate@5', 0.0)):>8} "
            f"{_format_pct(m.get('precision@5', 0.0)):>9} "
            f"{_format_delta(m.get('precision@5', 0.0) - pop_metrics.get('precision@5', 0.0)):>8} "
            f"{_format_pct(m.get('recall@5', 0.0)):>11} "
            f"{_format_delta(m.get('recall@5', 0.0) - pop_metrics.get('recall@5', 0.0)):>8} "
            f"{m.get('ndcg@5', 0.0):>9.4f} "
            f"{m.get('ndcg@5', 0.0) - pop_metrics.get('ndcg@5', 0.0):>+8.4f}"
        )


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
    summary_rows: list[dict] = []

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

        # ── Popularidad ──
        t0 = time.time()
        pop_metrics = evaluate_popularity_at_cutoff(
            orders_train, details_train, orders_test, details_test)
        pop_time = time.time() - t0

        svd_runs: list[dict] = []
        for n_components in SVD_COMPONENTS_GRID:
            t0 = time.time()
            svd_metrics = evaluate_svd_at_cutoff(
                orders_train,
                details_train,
                orders_test,
                details_test,
                n_components=n_components,
            )
            svd_time = time.time() - t0
            svd_runs.append(
                {
                    "requested_components": n_components,
                    "time_s": round(svd_time, 2),
                    "metrics": svd_metrics,
                }
            )
            summary_rows.append(
                {
                    "cutoff_pct": cutoff_pct,
                    "requested_components": n_components,
                    "actual_components": svd_metrics["actual_components"],
                    "hit_rate@5_pop": pop_metrics.get("hit_rate@5", 0.0),
                    "hit_rate@5_svd": svd_metrics.get("hit_rate@5", 0.0),
                    "precision@5_pop": pop_metrics.get("precision@5", 0.0),
                    "precision@5_svd": svd_metrics.get("precision@5", 0.0),
                    "recall@5_pop": pop_metrics.get("recall@5", 0.0),
                    "recall@5_svd": svd_metrics.get("recall@5", 0.0),
                    "ndcg@5_pop": pop_metrics.get("ndcg@5", 0.0),
                    "ndcg@5_svd": svd_metrics.get("ndcg@5", 0.0),
                    "map_pop": pop_metrics.get("MAP", 0.0),
                    "map_svd": svd_metrics.get("MAP", 0.0),
                    "mrr_pop": pop_metrics.get("MRR", 0.0),
                    "mrr_svd": svd_metrics.get("MRR", 0.0),
                    "variance_explained": svd_metrics.get("variance_explained", 0.0),
                    "svd_time_s": round(svd_time, 2),
                    "pop_time_s": round(pop_time, 2),
                }
            )

        svd_runs.sort(key=lambda run: run["metrics"].get("hit_rate@5", 0.0), reverse=True)

        print("\n  Baseline popularidad (Top-5 global excluyendo compras previas)")
        print(
            f"  Hit@5={_format_pct(pop_metrics.get('hit_rate@5', 0.0))}  "
            f"Prec@5={_format_pct(pop_metrics.get('precision@5', 0.0))}  "
            f"Recall@5={_format_pct(pop_metrics.get('recall@5', 0.0))}  "
            f"NDCG@5={pop_metrics.get('ndcg@5', 0.0):.4f}  "
            f"MAP={pop_metrics.get('MAP', 0.0):.4f}  "
            f"MRR={pop_metrics.get('MRR', 0.0):.4f}"
        )

        print_top5_comparison_table(pop_metrics, svd_runs)
        print_parameter_detail_table(pop_metrics, svd_runs)

        best_run = svd_runs[0]
        best_metrics = best_run["metrics"]
        print("\n  Mejor configuración SVD del corte")
        print(
            f"  n_components={best_metrics['actual_components']}  "
            f"Hit@5={_format_pct(best_metrics.get('hit_rate@5', 0.0))}  "
            f"mejora={_format_delta(best_metrics.get('hit_rate@5', 0.0) - pop_metrics.get('hit_rate@5', 0.0))} pts  "
            f"factor={_format_factor(best_metrics.get('hit_rate@5', 0.0), pop_metrics.get('hit_rate@5', 0.0))}  "
            f"var={best_metrics.get('variance_explained', 0.0):.4f}  "
            f"tiempo={best_run['time_s']:.2f}s"
        )

        all_results[f"cutoff_{cutoff_pct:.0%}"] = {
            "cutoff_pct": cutoff_pct,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "n_train": len(orders_train),
            "n_test": len(orders_test),
            "n_reincidentes": len(overlap),
            "popularity": pop_metrics,
            "pop_time_s": round(pop_time, 2),
            "svd_runs": svd_runs,
            "best_svd": best_run,
        }

    # ═══════════════════════════════════════════════════════════════
    # RESUMEN EJECUTIVO MULTI-CUTOFF
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("  📊 RESUMEN EJECUTIVO: Top-5 Popularidad vs Top-5 SVD")
    print(f"{'=' * 70}")

    print("\n  Mejor configuración por cutoff")
    print(
        f"  {'Cutoff':<8} {'Comp':>5} {'Hit@5 Pop':>11} {'Hit@5 SVD':>11} "
        f"{'Δ Hit':>8} {'Factor':>8} {'NDCG@5':>9} {'MAP':>8}"
    )
    print(f"  {'─' * 82}")
    for data in all_results.values():
        best = data["best_svd"]["metrics"]
        pop = data["popularity"]
        hit5_svd = best.get("hit_rate@5", 0.0)
        hit5_pop = pop.get("hit_rate@5", 0.0)
        cutoff_label = f"{data['cutoff_pct']:.0%}"
        print(
            f"  {cutoff_label:<8} "
            f"{best['actual_components']:>5} "
            f"{_format_pct(hit5_pop):>11} "
            f"{_format_pct(hit5_svd):>11} "
            f"{_format_delta(hit5_svd - hit5_pop):>8} "
            f"{_format_factor(hit5_svd, hit5_pop):>8} "
            f"{best.get('ndcg@5', 0.0):>9.4f} "
            f"{best.get('MAP', 0.0):>8.4f}"
        )

    print("\n  Todas las configuraciones evaluadas")
    print(
        f"  {'Cutoff':<8} {'Comp':>5} {'Hit@5 Pop':>11} {'Hit@5 SVD':>11} "
        f"{'Δ Hit':>8} {'Prec@5 SVD':>12} {'Recall@5 SVD':>14} {'Var':>7}"
    )
    print(f"  {'─' * 92}")
    for row in summary_rows:
        cutoff_label = f"{row['cutoff_pct']:.0%}"
        print(
            f"  {cutoff_label:<8} "
            f"{row['actual_components']:>5} "
            f"{_format_pct(row['hit_rate@5_pop']):>11} "
            f"{_format_pct(row['hit_rate@5_svd']):>11} "
            f"{_format_delta(row['hit_rate@5_svd'] - row['hit_rate@5_pop']):>8} "
            f"{_format_pct(row['precision@5_svd']):>12} "
            f"{_format_pct(row['recall@5_svd']):>14} "
            f"{row['variance_explained']:>7.3f}"
        )

    # Ganador global
    svd_wins = 0
    pop_wins = 0
    total_comparisons = 0
    for row in summary_rows:
        for metric in ["hit_rate@5", "precision@5", "recall@5", "ndcg@5", "map", "mrr"]:
            if metric == "map":
                s = row["map_svd"]
                p = row["map_pop"]
            elif metric == "mrr":
                s = row["mrr_svd"]
                p = row["mrr_pop"]
            else:
                s = row[f"{metric}_svd"]
                p = row[f"{metric}_pop"]
            total_comparisons += 1
            if s > p:
                svd_wins += 1
            elif p > s:
                pop_wins += 1

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
