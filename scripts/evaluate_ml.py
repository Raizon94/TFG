"""
scripts/evaluate_ml.py — Evaluación temporal: SVD vs Popularidad
=========================================================================
Compara la eficacia del modelo SVD contra un baseline de "productos más
vendidos", usando la misma metodología de corte temporal:

  1. Corta los datos en una fecha (ej. 80% histórico / 20% futuro)
  2. Entrena SVD SOLO con los datos históricos
  3. Calcula el ranking de popularidad SOLO con los datos históricos
  4. Para cada cliente que compra después del corte, genera recomendaciones
     con ambos métodos y compara con lo que REALMENTE compró

Métricas (para ambos modelos):
  • Hit Rate @K   — % de usuarios con ≥1 acierto en top K
  • Precision @K  — De las K recomendaciones, ¿cuántas se compraron?
  • Recall @K     — De los productos comprados, ¿cuántos estaban en top K?
  • NDCG @K       — Calidad del ranking (productos correctos más arriba = mejor)

Uso:
    python -m scripts.evaluate_ml                    # corte temporal por defecto (80%)
    python -m scripts.evaluate_ml --cutoff 0.7       # usar 70% como histórico
"""

from __future__ import annotations

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
RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation_results"

# K values para evaluar recomendaciones
K_VALUES = [3, 5, 10]


# ═══════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════

def ndcg_at_k(recommended: list[int], actual: set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Mide no solo si acertamos, sino si los aciertos están en las
    primeras posiciones (más arriba en el ranking = más valor).
    Rango: 0.0 (ningún acierto) a 1.0 (perfecto).
    """
    dcg = 0.0
    for i, pid in enumerate(recommended[:k]):
        if pid in actual:
            # +1 porque el índice empieza en 0
            dcg += 1.0 / np.log2(i + 2)

    # DCG ideal: todos los aciertos posibles en las primeras posiciones
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════

def load_data(engine):
    """Carga todos los pedidos y detalles de la BD."""
    orders_df = pd.read_sql(
        """
        SELECT o.id_order, o.id_customer, o.date_add, o.total_paid_real
        FROM ps_orders o
        WHERE o.valid = 1
        ORDER BY o.date_add
        """,
        engine,
        parse_dates=["date_add"],
    )

    details_df = pd.read_sql(
        """
        SELECT od.id_order, od.product_id, od.product_quantity
        FROM ps_order_detail od
        JOIN ps_orders o ON od.id_order = o.id_order
        WHERE o.valid = 1
        """,
        engine,
    )

    return orders_df, details_df


# ═══════════════════════════════════════════════════════════════════
# ENTRENAMIENTO + ARTEFACTOS
# ═══════════════════════════════════════════════════════════════════

def build_training_artifacts(orders_train: pd.DataFrame, details_train: pd.DataFrame):
    """
    Entrena SVD con datos históricos y devuelve artefactos necesarios
    para evaluación (incluye fechas para decaimiento temporal).
    """
    merged_train = orders_train[["id_order", "id_customer", "date_add"]].merge(
        details_train[["id_order", "product_id", "product_quantity"]],
        on="id_order",
    )
    interaction = (
        merged_train
        .groupby(["id_customer", "product_id"], as_index=False)
        .agg(qty=("product_quantity", "sum"), last_date=("date_add", "max"))
    )

    customer_ids = sorted(interaction["id_customer"].unique())
    product_ids = sorted(interaction["product_id"].unique())

    cust_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
    prod_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    idx_to_prod = {i: pid for pid, i in prod_to_idx.items()}

    rows = interaction["id_customer"].map(cust_to_idx).values
    cols = interaction["product_id"].map(prod_to_idx).values
    vals = np.log1p(interaction["qty"].values.astype(float))

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(customer_ids), len(product_ids))
    )

    n_components = min(SVD_N_COMPONENTS, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    svd.fit(matrix)

    print(f"  SVD entrenado: {len(customer_ids):,} usuarios × {len(product_ids):,} productos")
    print(f"  Varianza explicada: {svd.explained_variance_ratio_.sum():.2%}")

    return svd, matrix, cust_to_idx, prod_to_idx, idx_to_prod, interaction


# ═══════════════════════════════════════════════════════════════════
# EVALUACIÓN DEL MODELO DE RECOMENDACIONES (SVD)
# ═══════════════════════════════════════════════════════════════════

def evaluate_what(orders_train: pd.DataFrame, details_train: pd.DataFrame,
                  orders_test: pd.DataFrame, details_test: pd.DataFrame) -> dict:
    """
    Evalúa el modelo "Qué" con corte temporal:

    1. Entrena SVD con compras ANTES del corte
    2. Para cada cliente que compró DESPUÉS del corte:
       - Usa fold-in con su historial pre-corte
       - Genera top K recomendaciones
       - Compara con los productos que REALMENTE compró post-corte
    """
    print("\n" + "─" * 60)
    print("  EVALUACIÓN: Modelo 'Qué' (Recomendación de Productos)")
    print("─" * 60)

    # ── Entrenar SVD con datos históricos ──
    svd, matrix, cust_to_idx, prod_to_idx, idx_to_prod, interaction = (
        build_training_artifacts(orders_train, details_train)
    )

    # ── Ground truth: ¿qué compró cada cliente DESPUÉS del corte? ──
    test_order_ids = set(orders_test["id_order"])
    test_details = details_test[details_test["id_order"].isin(test_order_ids)]
    ground_truth = (
        test_details
        .groupby(orders_test.set_index("id_order").loc[
            test_details["id_order"], "id_customer"
        ].values)["product_id"]
        .apply(set)
        .to_dict()
    )

    # Simplificamos: agrupar por customer directamente
    merged_test = orders_test[["id_order", "id_customer"]].merge(
        test_details[["id_order", "product_id"]],
        on="id_order",
    )
    ground_truth = merged_test.groupby("id_customer")["product_id"].apply(set).to_dict()

    # Solo evaluar clientes que existían en el historial
    eval_customers = [c for c in ground_truth if c in cust_to_idx]
    print(f"  Clientes evaluables (tienen historial + compras futuras): {len(eval_customers):,}")

    if not eval_customers:
        print("  No hay clientes evaluables")
        return {}

    # ── Para cada cliente: fold-in → top K → comparar ──
    results_by_k = {k: {"hits": 0, "precision_sum": 0, "recall_sum": 0,
                         "ndcg_sum": 0} for k in K_VALUES}
    n_eval = 0

    for cust_id in eval_customers:
        actual_products = ground_truth[cust_id]
        # Solo contar productos que el SVD conoce
        actual_known = actual_products.intersection(prod_to_idx.keys())
        if not actual_known:
            continue

        # Fold-in: construir vector con historial pre-corte
        user_idx = cust_to_idx[cust_id]
        user_vec = matrix[user_idx].toarray().reshape(1, -1)

        if user_vec.sum() == 0:
            continue

        user_latent = svd.transform(user_vec)
        scores = (user_latent @ svd.components_).flatten()

        # Excluir productos ya comprados en historial
        purchased = matrix[user_idx].toarray().flatten()
        scores[purchased > 0] = -np.inf

        # Top recomendaciones
        max_k = max(K_VALUES)
        top_indices = np.argsort(scores)[::-1][:max_k]
        recommended = [idx_to_prod[int(i)] for i in top_indices]

        n_eval += 1

        for k in K_VALUES:
            top_k = recommended[:k]
            top_k_set = set(top_k)

            hits = len(top_k_set & actual_known)
            results_by_k[k]["hits"] += 1 if hits > 0 else 0
            results_by_k[k]["precision_sum"] += hits / k
            results_by_k[k]["recall_sum"] += hits / len(actual_known) if actual_known else 0
            results_by_k[k]["ndcg_sum"] += ndcg_at_k(top_k, actual_known, k)

    # ── Calcular métricas finales ──
    metrics = {}
    print(f"\n  Clientes evaluados (con historial SVD válido): {n_eval:,}")
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  K │ Hit Rate │ Precision │ Recall  │  NDCG   │            │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")

    for k in K_VALUES:
        r = results_by_k[k]
        hit_rate = r["hits"] / n_eval * 100 if n_eval else 0
        precision = r["precision_sum"] / n_eval * 100 if n_eval else 0
        recall = r["recall_sum"] / n_eval * 100 if n_eval else 0
        ndcg = r["ndcg_sum"] / n_eval if n_eval else 0

        metrics[f"hit_rate@{k}"] = hit_rate
        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = ndcg

        print(f"  │ {k:>2} │  {hit_rate:>5.1f}%  │   {precision:>5.1f}%  │ {recall:>5.1f}%  │ {ndcg:>5.3f}  │            │")

    print(f"  └─────────────────────────────────────────────────────────────┘")

    metrics["n_eval"] = n_eval
    return metrics




# ═══════════════════════════════════════════════════════════════════
# BASELINE: PRODUCTOS MÁS VENDIDOS
# ═══════════════════════════════════════════════════════════════════

def evaluate_popularity_baseline(
    orders_train: pd.DataFrame, details_train: pd.DataFrame,
    orders_test: pd.DataFrame, details_test: pd.DataFrame,
) -> dict:
    """
    Baseline: recomendar simplemente los productos más vendidos del
    período de entrenamiento (excluyendo los que cada cliente ya compró).
    """
    print("\n" + "─" * 60)
    print("  BASELINE: Productos más vendidos (Popularidad)")
    print("─" * 60)

    # ── Ranking de popularidad del período de entrenamiento ──
    merged_train = orders_train[["id_order", "id_customer"]].merge(
        details_train[["id_order", "product_id", "product_quantity"]],
        on="id_order",
    )
    popularity = (
        merged_train.groupby("product_id")["product_quantity"]
        .sum()
        .sort_values(ascending=False)
    )
    popular_ranking = list(popularity.index)  # lista ordenada por ventas
    print(f"  Productos en ranking de popularidad: {len(popular_ranking):,}")
    print(f"  Top 5: {popular_ranking[:5]}")

    # ── Historial de compra por cliente (para excluir) ──
    customer_history = (
        merged_train.groupby("id_customer")["product_id"].apply(set).to_dict()
    )

    # ── Ground truth: ¿qué compró cada cliente post-corte? ──
    merged_test = orders_test[["id_order", "id_customer"]].merge(
        details_test[["id_order", "product_id"]],
        on="id_order",
    )
    ground_truth = merged_test.groupby("id_customer")["product_id"].apply(set).to_dict()

    # Solo clientes con historial previo (misma condición que SVD)
    eval_customers = [c for c in ground_truth if c in customer_history]
    print(f"  Clientes evaluables: {len(eval_customers):,}")

    if not eval_customers:
        print("  No hay clientes evaluables")
        return {}

    # ── Para cada cliente: recomendar populares (sin los que ya compró) ──
    results_by_k = {k: {"hits": 0, "precision_sum": 0, "recall_sum": 0,
                         "ndcg_sum": 0} for k in K_VALUES}
    n_eval = 0
    max_k = max(K_VALUES)

    for cust_id in eval_customers:
        actual_products = ground_truth[cust_id]
        if not actual_products:
            continue

        already_bought = customer_history.get(cust_id, set())
        recommended = [p for p in popular_ranking if p not in already_bought][:max_k]

        if not recommended:
            continue

        n_eval += 1

        for k in K_VALUES:
            top_k = recommended[:k]
            top_k_set = set(top_k)

            hits = len(top_k_set & actual_products)
            results_by_k[k]["hits"] += 1 if hits > 0 else 0
            results_by_k[k]["precision_sum"] += hits / k
            results_by_k[k]["recall_sum"] += hits / len(actual_products) if actual_products else 0
            results_by_k[k]["ndcg_sum"] += ndcg_at_k(top_k, actual_products, k)

    # ── Métricas finales ──
    metrics = {}
    print(f"\n  Clientes evaluados: {n_eval:,}")
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  K │ Hit Rate │ Precision │ Recall  │  NDCG   │            │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")

    for k in K_VALUES:
        r = results_by_k[k]
        hit_rate = r["hits"] / n_eval * 100 if n_eval else 0
        precision = r["precision_sum"] / n_eval * 100 if n_eval else 0
        recall = r["recall_sum"] / n_eval * 100 if n_eval else 0
        ndcg = r["ndcg_sum"] / n_eval if n_eval else 0

        metrics[f"hit_rate@{k}"] = hit_rate
        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = ndcg

        print(f"  │ {k:>2} │  {hit_rate:>5.1f}%  │   {precision:>5.1f}%  │ {recall:>5.1f}%  │ {ndcg:>5.3f}  │            │")

    print(f"  └─────────────────────────────────────────────────────────────┘")

    metrics["n_eval"] = n_eval
    return metrics


# ═══════════════════════════════════════════════════════════════════
# VISUALIZACIÓN DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════

def generate_charts(svd_results: dict, pop_results: dict):
    """Genera gráficos comparativos SVD vs Popularidad."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib no disponible. Instala con: pip install matplotlib")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not svd_results or not pop_results:
        return

    # ── Gráfico 1: Hit Rate comparativo ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1a. Hit Rate SVD vs Popularidad
    ax = axes[0]
    x = np.arange(len(K_VALUES))
    width = 0.35
    svd_hr = [svd_results.get(f"hit_rate@{k}", 0) for k in K_VALUES]
    pop_hr = [pop_results.get(f"hit_rate@{k}", 0) for k in K_VALUES]

    bars1 = ax.bar(x - width/2, svd_hr, width, label="SVD (ML)", color="#2196F3")
    bars2 = ax.bar(x + width/2, pop_hr, width, label="Popularidad", color="#FF5722")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.set_ylabel("Hit Rate (%)", fontsize=11)
    ax.set_title("Hit Rate @ K", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # 1b. Precision SVD vs Popularidad
    ax = axes[1]
    svd_prec = [svd_results.get(f"precision@{k}", 0) for k in K_VALUES]
    pop_prec = [pop_results.get(f"precision@{k}", 0) for k in K_VALUES]

    bars1 = ax.bar(x - width/2, svd_prec, width, label="SVD (ML)", color="#4CAF50")
    bars2 = ax.bar(x + width/2, pop_prec, width, label="Popularidad", color="#FF9800")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.set_ylabel("Precision (%)", fontsize=11)
    ax.set_title("Precision @ K", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # 1c. NDCG SVD vs Popularidad
    ax = axes[2]
    svd_ndcg = [svd_results.get(f"ndcg@{k}", 0) for k in K_VALUES]
    pop_ndcg = [pop_results.get(f"ndcg@{k}", 0) for k in K_VALUES]

    bars1 = ax.bar(x - width/2, svd_ndcg, width, label="SVD (ML)", color="#9C27B0")
    bars2 = ax.bar(x + width/2, pop_ndcg, width, label="Popularidad", color="#795548")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.set_ylabel("NDCG", fontsize=11)
    ax.set_title("NDCG @ K", fontsize=13, fontweight="bold")
    all_ndcg = svd_ndcg + pop_ndcg
    ax.set_ylim(0, max(all_ndcg) * 1.3 if all_ndcg and max(all_ndcg) > 0 else 1)
    ax.legend(fontsize=11)

    plt.suptitle("SVD (Machine Learning)  vs  Productos más vendidos (Baseline)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "comparativa_svd_vs_popularidad.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n Gráfico comparativo guardado: {path}")

    # ── Gráfico 2: Tabla resumen con mejora porcentual ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    headers = ["Métrica"] + [f"K={k}" for k in K_VALUES]
    rows = []
    for metric_name, label in [("hit_rate", "Hit Rate"), ("precision", "Precision"),
                                ("recall", "Recall"), ("ndcg", "NDCG")]:
        row = [label]
        for k in K_VALUES:
            s = svd_results.get(f"{metric_name}@{k}", 0)
            p = pop_results.get(f"{metric_name}@{k}", 0)
            if metric_name == "ndcg":
                diff = s - p
                sign = "+" if diff >= 0 else ""
                row.append(f"SVD {s:.3f} vs Pop {p:.3f}\n({sign}{diff:.3f})")
            else:
                diff = s - p
                sign = "+" if diff >= 0 else ""
                row.append(f"SVD {s:.1f}% vs Pop {p:.1f}%\n({sign}{diff:.1f}pp)")
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colWidths=[0.12] + [0.29] * len(K_VALUES))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Colorear celdas según quién gana
    for i, row in enumerate(rows):
        for j, k in enumerate(K_VALUES):
            metric_name = ["hit_rate", "precision", "recall", "ndcg"][i]
            s = svd_results.get(f"{metric_name}@{k}", 0)
            p = pop_results.get(f"{metric_name}@{k}", 0)
            cell = table[i + 1, j + 1]
            if s > p:
                cell.set_facecolor("#E8F5E9")  # verde claro: SVD gana
            elif p > s:
                cell.set_facecolor("#FFEBEE")  # rojo claro: Popularidad gana
            else:
                cell.set_facecolor("#FFF9C4")  # amarillo: empate

    # Header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("Comparativa detallada: SVD vs Popularidad\n(verde = SVD gana, rojo = Popularidad gana)",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = RESULTS_DIR / "tabla_comparativa.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Tabla comparativa guardada: {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main(cutoff_pct: float = 0.80) -> None:
    """
    Ejecuta la evaluación temporal completa.

    Args:
        cutoff_pct: porcentaje de pedidos (por fecha) para entrenamiento.
                    El resto se usa como test. Default: 80%.
    """
    print("=" * 60)
    print("  Evaluación Temporal del Motor de IA")
    print("=" * 60)

    engine = create_engine(DATABASE_URL, echo=False)
    orders_df, details_df = load_data(engine)

    # ── Corte temporal ──
    orders_sorted = orders_df.sort_values("date_add")
    cutoff_idx = int(len(orders_sorted) * cutoff_pct)
    cutoff_date = orders_sorted.iloc[cutoff_idx]["date_add"]

    orders_train = orders_sorted[orders_sorted["date_add"] <= cutoff_date].copy()
    orders_test = orders_sorted[orders_sorted["date_add"] > cutoff_date].copy()

    # Detalles correspondientes
    train_order_ids = set(orders_train["id_order"])
    test_order_ids = set(orders_test["id_order"])
    details_train = details_df[details_df["id_order"].isin(train_order_ids)].copy()
    details_test = details_df[details_df["id_order"].isin(test_order_ids)].copy()

    print(f"\n  Fecha de corte: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  ┌────────────────────────────────────────────┐")
    print(f"  │  Conjunto  │ Pedidos │ Clientes │ Período  │")
    print(f"  ├────────────────────────────────────────────┤")
    print(f"  │  TRAIN     │ {len(orders_train):>7,} │ {orders_train['id_customer'].nunique():>8,} │ {orders_train['date_add'].min().strftime('%Y-%m')} → {orders_train['date_add'].max().strftime('%Y-%m')} │")
    print(f"  │  TEST      │ {len(orders_test):>7,} │ {orders_test['id_customer'].nunique():>8,} │ {orders_test['date_add'].min().strftime('%Y-%m')} → {orders_test['date_add'].max().strftime('%Y-%m')} │")
    print(f"  └────────────────────────────────────────────┘")

    # Clientes que aparecen en ambos (reincidentes = el caso interesante)
    train_customers = set(orders_train["id_customer"])
    test_customers = set(orders_test["id_customer"])
    overlap = train_customers & test_customers
    new_in_test = test_customers - train_customers
    print(f"\n  Clientes reincidentes (en train Y test): {len(overlap):,}")
    print(f"  Clientes nuevos (solo en test):          {len(new_in_test):,}")

    # ── Evaluar SVD ──
    svd_results = evaluate_what(orders_train, details_train,
                                orders_test, details_test)



    # ── Evaluar baseline de popularidad ──
    pop_results = evaluate_popularity_baseline(orders_train, details_train,
                                               orders_test, details_test)

    # ── Generar gráficos comparativos ──
    generate_charts(svd_results, pop_results)

    # ── Resumen comparativo final ──
    print("\n" + "=" * 70)
    print(" COMPARATIVA FINAL: SVD (ML) vs Popularidad (Baseline)")
    print("=" * 70)

    if svd_results and pop_results:
        print(f"\n  {'Métrica':<20} {'SVD (ML)':>12} {'Popular':>12} {'Δ':>10} {'Ganador':>10}")
        print(f"  {'─' * 66}")

        for k in K_VALUES:
            print(f"\n  --- K = {k} ---")
            for metric, label, fmt in [
                ("hit_rate", "Hit Rate", ".1f"),
                ("precision", "Precision", ".1f"),
                ("recall", "Recall", ".1f"),
                ("ndcg", "NDCG", ".3f"),
            ]:
                s = svd_results.get(f"{metric}@{k}", 0)
                p = pop_results.get(f"{metric}@{k}", 0)
                diff = s - p
                if metric == "ndcg":
                    winner = "🟢 SVD" if s > p else ("🔴 POP" if p > s else "🟡 EMPATE")
                    sign = "+" if diff >= 0 else ""
                    print(f"  {label + ' @' + str(k):<20} {s:>11{fmt}} {p:>11{fmt}} {sign}{diff:>9{fmt}} {winner:>10}")
                else:
                    winner = "🟢 SVD" if s > p else ("🔴 POP" if p > s else "🟡 EMPATE")
                    sign = "+" if diff >= 0 else ""
                    print(f"  {label + ' @' + str(k):<20} {s:>10{fmt}}% {p:>10{fmt}}% {sign}{diff:>8{fmt}}pp {winner:>10}")

        # Resumen ejecutivo
        svd_wins = 0
        pop_wins = 0
        for k in K_VALUES:
            for metric in ["hit_rate", "precision", "recall", "ndcg"]:
                s = svd_results.get(f"{metric}@{k}", 0)
                p = pop_results.get(f"{metric}@{k}", 0)
                if s > p:
                    svd_wins += 1
                elif p > s:
                    pop_wins += 1

        total = svd_wins + pop_wins
        print(f"\n  {'=' * 66}")
        print(f" MARCADOR: SVD gana en {svd_wins}/{total} métricas, "
              f"Popularidad gana en {pop_wins}/{total}")

        # Mejora media en Hit Rate
        avg_improvement = np.mean([
            svd_results.get(f"hit_rate@{k}", 0) - pop_results.get(f"hit_rate@{k}", 0)
            for k in K_VALUES
        ])
        if avg_improvement > 0:
            print(f" SVD supera a Popularidad en Hit Rate por +{avg_improvement:.1f}pp de media")
        elif avg_improvement < 0:
            print(f"  Popularidad supera a SVD en Hit Rate por {-avg_improvement:.1f}pp de media")
        else:
            print(f" Ambos modelos empatan en Hit Rate")

    print(f"\n Gráficos en: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluación temporal del motor ML")
    parser.add_argument("--cutoff", type=float, default=0.80,
                        help="Porcentaje de datos para entrenamiento (default: 0.80)")
    args = parser.parse_args()
    main(cutoff_pct=args.cutoff)
