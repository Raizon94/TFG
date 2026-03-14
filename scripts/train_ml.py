"""
scripts/train_ml.py — Entrenamiento del motor de recomendaciones
=================================================================
Genera el modelo de Filtrado Colaborativo basado en TruncatedSVD:
  • Construye una matriz usuario-producto con datos reales de la BD
  • Descompone con SVD para encontrar patrones de co-compra
  • Guarda los artefactos .joblib en backend/models/

Se conecta a la BD real de PrestaShop y guarda los artefactos .joblib
en backend/models/ para ser consumidos por la API en tiempo real.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sqlalchemy import create_engine

# ─── Configuración ──────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:root1234@localhost:3306/tfg_bd",
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"

RANDOM_STATE = 42

# Parámetros del modelo SVD
# n_components = número de factores latentes.
# Concepto: cada "componente" captura un patrón de co-compra oculto
# (ej. "clientes que compran cuarzos también compran pulseras").
# 50 componentes es un buen balance entre expresividad y overfitting
# para catálogos de ~2000 productos.
SVD_N_COMPONENTS = 50

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    print("=" * 60)
    print("  Motor de Recomendaciones — Entrenamiento")
    print("=" * 60)

    # Crear directorio de modelos si no existe
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DATABASE_URL, echo=False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PASO 1: Extracción de datos
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[1/3] Extrayendo datos de la BD...")

    # Pedidos válidos con fecha
    orders_df = pd.read_sql(
        """
        SELECT o.id_order, o.id_customer, o.date_add, o.total_paid_real
        FROM ps_orders o
        WHERE o.valid = 1
        ORDER BY o.id_customer, o.date_add
        """,
        engine,
        parse_dates=["date_add"],
    )

    # Detalle de cada pedido (qué productos y cuántos)
    details_df = pd.read_sql(
        """
        SELECT od.id_order, od.product_id, od.product_quantity
        FROM ps_order_detail od
        JOIN ps_orders o ON od.id_order = o.id_order
        WHERE o.valid = 1
        """,
        engine,
    )

    n_orders = len(orders_df)
    n_customers = orders_df["id_customer"].nunique()
    n_products = details_df["product_id"].nunique()
    print(f"    → {n_orders:,} pedidos de {n_customers:,} clientes sobre {n_products:,} productos")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PASO 2: Ingeniería de la matriz usuario-producto
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[2/3] Construyendo matriz usuario-producto y entrenando SVD...")

    # Unir pedidos con detalles para tener (id_customer, product_id, qty)
    merged = orders_df[["id_order", "id_customer"]].merge(
        details_df[["id_order", "product_id", "product_quantity"]],
        on="id_order",
    )

    # Matriz de interacción: filas = clientes, columnas = productos
    # Valores = suma total de unidades compradas de ese producto.
    # Esta es una "señal implícita de preferencia" (más cantidad = más interés).
    interaction = merged.groupby(["id_customer", "product_id"])["product_quantity"].sum().reset_index()
    interaction.columns = ["id_customer", "product_id", "qty"]

    # Crear mapeos de IDs originales ↔ índices de la matriz
    # (los IDs de PrestaShop no son consecutivos, necesitamos reindexar)
    customer_ids = sorted(interaction["id_customer"].unique())
    product_ids = sorted(interaction["product_id"].unique())

    cust_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
    prod_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    idx_to_prod = {i: pid for pid, i in prod_to_idx.items()}

    # Construir matriz dispersa (sparse) para eficiencia de memoria
    rows = interaction["id_customer"].map(cust_to_idx).values
    cols = interaction["product_id"].map(prod_to_idx).values
    vals = interaction["qty"].values.astype(float)

    # Normalizar con log(1 + x) para suavizar clientes que compran
    # cantidades enormes y evitar que dominen la descomposición.
    vals_log = np.log1p(vals)

    user_product_matrix = csr_matrix(
        (vals_log, (rows, cols)),
        shape=(len(customer_ids), len(product_ids))
    )

    print(f"    → Matriz dispersa: {user_product_matrix.shape[0]:,} usuarios × {user_product_matrix.shape[1]:,} productos")
    print(f"    → Densidad: {user_product_matrix.nnz / (user_product_matrix.shape[0] * user_product_matrix.shape[1]):.4%}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PASO 3: TruncatedSVD — Filtrado Colaborativo
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TruncatedSVD descompone la matriz M ≈ U·Σ·Vᵀ
    # donde cada fila de U·Σ es el "perfil latente" del usuario
    # y cada columna de Vᵀ es el "perfil latente" del producto.
    # Al reconstruir M̂ = U·Σ·Vᵀ, las celdas que eran 0 (productos
    # no comprados) ahora tienen un score > 0 que indica la afinidad
    # predicha del usuario con ese producto.

    # Ajustar n_components al mínimo entre lo deseado y lo posible
    n_components = min(SVD_N_COMPONENTS, min(user_product_matrix.shape) - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(user_product_matrix)

    explained_var = svd.explained_variance_ratio_.sum()
    print(f"    → SVD con {n_components} componentes, varianza explicada: {explained_var:.2%}")

    # Guardar todo lo necesario para reconstruir recomendaciones en runtime
    what_artifact = {
        "svd_model": svd,
        "user_factors": user_factors,           # U·Σ  (n_users × n_components)
        "user_product_matrix": user_product_matrix,
        "customer_ids": customer_ids,            # mapeo idx → id_customer
        "product_ids": product_ids,              # mapeo idx → product_id
        "cust_to_idx": cust_to_idx,             # id_customer → idx
        "prod_to_idx": prod_to_idx,             # product_id → idx
        "idx_to_prod": idx_to_prod,             # idx → product_id
    }
    what_path = MODELS_DIR / "svd_what.joblib"
    joblib.dump(what_artifact, what_path)
    print(f"    → Modelo guardado en: {what_path}")

    # ─── Productos más populares (fallback para usuarios nuevos) ──
    # Si un usuario no tiene historial, recomendamos los más vendidos.
    top_products = (
        interaction.groupby("product_id")["qty"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )
    fallback_path = MODELS_DIR / "fallback_popular.joblib"
    joblib.dump(top_products, fallback_path)
    print(f"    → Productos populares (fallback): {top_products[:5]}...")

    # ─── Metadatos de entrenamiento (para reentrenamiento inteligente) ──
    # Estos metadatos permiten al sistema saber cuándo reentrenar:
    # si el número de pedidos actuales supera order_count + umbral,
    # se lanza un reentrenamiento automático en segundo plano.
    from datetime import datetime as _dt
    training_metadata = {
        "order_count": n_orders,
        "customer_count": n_customers,
        "product_count": n_products,
        "trained_at": _dt.now().isoformat(),
        "metrics": {"svd_variance": explained_var},
    }
    meta_path = MODELS_DIR / "training_metadata.joblib"
    joblib.dump(training_metadata, meta_path)
    print(f"    → Metadatos guardados en: {meta_path}")

    # ─── Resumen final ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Entrenamiento completado con exito")
    print("=" * 60)
    print(f"  Artefactos generados en: {MODELS_DIR}/")
    print(f"    • svd_what.joblib      ({what_path.stat().st_size / 1024:.0f} KB)")
    print(f"    • fallback_popular.joblib")
    print(f"\n  Métricas:")
    print(f"    • Varianza explicada SVD: {explained_var:.2%}")
    print(f"    • {n_components} factores latentes sobre {len(product_ids):,} productos")
    print("=" * 60)


if __name__ == "__main__":
    main()
