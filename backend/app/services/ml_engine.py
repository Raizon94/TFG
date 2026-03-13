"""
backend/ml_engine.py — Motor de IA en TIEMPO REAL
==================================================
Separa INFERENCIA (instantánea) de ENTRENAMIENTO (periódico):

  • Interacciones del usuario → calculadas LIVE con SQL en cada petición
  • Pesos del modelo (SVD)    → cargados desde .joblib, reentrenados
                                 solo cuando hay suficientes datos nuevos

Esto significa que CADA NUEVA COMPRA se refleja inmediatamente
en las recomendaciones, sin necesidad de reentrenar.

Funciones principales:
  predict_what()  → recomendaciones con fold-in SVD live
  check_retrain_needed() → ¿hay suficientes datos nuevos?
  trigger_retrain_async() → lanza reentrenamiento en segundo plano
"""

from __future__ import annotations

import io
import logging
import threading
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("uvicorn.error")

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Umbral: reentrenar tras N pedidos nuevos desde el último entrenamiento.
# En producción real esto sería configurable (env var, panel admin, etc.).
RETRAIN_THRESHOLD = 50

# Lock para evitar reentrenamientos simultáneos
_retrain_lock = threading.Lock()
_is_retraining = False
_retrain_log_text = ""


# ═══════════════════════════════════════════════════════════════════
# 1. PREDICCIÓN "QUÉ" — Fold-in SVD con compras live
# ═══════════════════════════════════════════════════════════════════

def predict_what(user_id: int, db: Session, svd_artifact: dict) -> list[int]:
    """
    Recomienda productos usando las compras REALES ACTUALES del usuario,
    proyectadas al espacio latente del SVD mediante "fold-in".

    Técnica Fold-in con VENTANA DESLIZANTE:
      1. Lee las compras actuales del usuario desde ps_order_detail
      2. Usa solo los últimos MAX_WINDOW productos distintos (por fecha),
         para que las compras recientes tengan peso relativo significativo
      3. Construye su vector de interacción x (1 × n_products) con
         codificación BINARIA (1.0 si comprado, 0 si no) para que cada
         producto en la ventana tenga el mismo peso
      4. Proyecta al espacio latente:  u = x · Vᵀ     (transform)
      5. Reconstruye scores:           s = u · V       (n_products,)
      6. Excluye solo los productos de la ventana (compras recientes),
         permitiendo que compras antiguas reaparezcan como recomendación
      7. Devuelve los top 10 por score

    Ventaja: al usar ventana pequeña + codificación binaria, una sola
    compra nueva cambia ~12% del vector → cambio visible en las
    recomendaciones. Con log1p(qty) y ventana=20, una compra nueva
    apenas cambiaba el 2-3% del vector → cambio imperceptible.

    Args:
        user_id: ID del cliente
        db: sesión SQLAlchemy
        svd_artifact: dict con "svd_model", "product_ids", "idx_to_prod"

    Returns:
        Lista de hasta 10 product_ids recomendados.
    """
    product_ids = svd_artifact["product_ids"]
    idx_to_prod = svd_artifact["idx_to_prod"]
    svd_model = svd_artifact["svd_model"]

    # Mapeo producto → índice en la matriz SVD
    prod_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    # ── Consultar compras actuales con fecha para ventana deslizante ──
    rows = db.execute(text("""
        SELECT od.product_id,
               SUM(od.product_quantity) AS qty,
               MAX(o.date_add)          AS last_date
        FROM ps_order_detail od
        JOIN ps_orders o ON od.id_order = o.id_order
        WHERE o.id_customer = :uid AND o.valid = 1
        GROUP BY od.product_id
        ORDER BY last_date DESC
    """), {"uid": user_id}).fetchall()

    if not rows:
        return []

    # ── Ventana deslizante: solo los últimos N productos para el vector ──
    # Con ventana=8 y codificación binaria, cada nueva compra supone
    # ~12% del vector → cambio visible en las recomendaciones.
    MAX_WINDOW = 8

    n_products = len(product_ids)
    interaction_vec = np.zeros((1, n_products))
    window_mask = np.zeros(n_products, dtype=bool)

    window_count = 0
    for row in rows:
        pid = row.product_id
        if pid in prod_to_idx:
            idx = prod_to_idx[pid]
            if window_count < MAX_WINDOW:
                # Solo los productos de la ventana se usan para fold-in
                # Y solo estos se excluyen de las recomendaciones
                interaction_vec[0, idx] = 1.0
                window_mask[idx] = True
                window_count += 1

    if interaction_vec.sum() == 0:
        return []

    # ── Fold-in: proyectar al espacio latente del SVD ──
    # svd.transform(X) calcula X @ V (donde V = components_.T)
    user_latent = svd_model.transform(interaction_vec)  # (1, n_components)

    # Reconstruir scores: user_latent @ V^T → (1, n_products)
    scores = (user_latent @ svd_model.components_).flatten()

    # Excluir solo los productos de la ventana (las compras recientes).
    # Productos comprados hace tiempo SÍ pueden reaparecer como recomendación.
    scores[window_mask] = -np.inf

    # Top 10 por score de afinidad
    top_indices = np.argsort(scores)[::-1][:10]
    return [idx_to_prod[int(i)] for i in top_indices]


# ═══════════════════════════════════════════════════════════════════
# 2. REENTRENAMIENTO INTELIGENTE
# ═══════════════════════════════════════════════════════════════════

def get_training_metadata() -> dict | None:
    """Lee los metadatos del último entrenamiento desde disco."""
    meta_path = MODELS_DIR / "training_metadata.joblib"
    if meta_path.exists():
        return joblib.load(meta_path)
    return None


def check_retrain_needed(db: Session) -> tuple[bool, str]:
    """
    Comprueba si hay suficientes pedidos nuevos para justificar
    un reentrenamiento completo del modelo.

    Compara el número de pedidos actual con el que había cuando
    se entrenó por última vez. Si la diferencia supera RETRAIN_THRESHOLD,
    indica que hay que reentrenar.

    Returns:
        (should_retrain: bool, reason: str)
    """
    meta = get_training_metadata()

    if meta is None:
        return True, "No hay metadatos de entrenamiento previo"

    # Contar pedidos válidos actuales
    result = db.execute(
        text("SELECT COUNT(*) AS cnt FROM ps_orders WHERE valid = 1")
    ).fetchone()
    current_orders = result.cnt

    last_orders = meta.get("order_count", 0)
    new_orders = current_orders - last_orders

    if new_orders >= RETRAIN_THRESHOLD:
        return True, (
            f"{new_orders} pedidos nuevos desde \u00faltimo entrenamiento "
            f"(umbral: {RETRAIN_THRESHOLD})"
        )

    return False, (
        f"Solo {new_orders}/{RETRAIN_THRESHOLD} pedidos nuevos necesarios"
    )


def trigger_retrain_async(reload_callback=None) -> bool:
    """
    Lanza el reentrenamiento en un hilo de fondo (no bloquea la API).

    Args:
        reload_callback: función sin args a llamar tras completar
                         el reentrenamiento (típicamente _reload_models).

    Returns:
        True si se inició el reentrenamiento.
        False si ya había uno en curso.
    """
    global _is_retraining

    if _is_retraining:
        logger.info("\u23f3 Reentrenamiento ya en curso, ignorando petici\u00f3n.")
        return False

    def _do_retrain():
        global _is_retraining
        global _retrain_log_text
        with _retrain_lock:
            _is_retraining = True
            try:
                logger.info("\U0001f504 Iniciando reentrenamiento de modelos ML...")
                buffer = io.StringIO()
                buffer.write("\n" + "=" * 60 + "\n")
                buffer.write(f"Reentrenamiento iniciado: {datetime.now().isoformat()}\n")
                buffer.write("=" * 60 + "\n")

                from scripts.train_ml import main as train_main
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    train_main()

                buffer.write(f"\nReentrenamiento finalizado: {datetime.now().isoformat()}\n")
                buffer.write("=" * 60 + "\n")
                _retrain_log_text = buffer.getvalue()

                logger.info("\u2705 Reentrenamiento completado")

                if reload_callback:
                    reload_callback()
                    logger.info("\U0001f504 Modelos recargados en memoria")

            except Exception as e:
                logger.error("\u274c Error en reentrenamiento: %s", e, exc_info=True)
                _retrain_log_text = (
                    _retrain_log_text
                    + f"\n\u274c Error en reentrenamiento: {e}\n"
                )
            finally:
                _is_retraining = False

    thread = threading.Thread(target=_do_retrain, daemon=True, name="ml-retrain")
    thread.start()
    return True


def is_retraining() -> bool:
    """Devuelve True si hay un reentrenamiento en curso."""
    return _is_retraining


def get_retrain_log(lines: int = 200) -> str:
    """Devuelve las últimas líneas del log en memoria."""
    if not _retrain_log_text:
        return "No hay log de reentrenamiento aún."
    if lines <= 0:
        return _retrain_log_text
    parts = _retrain_log_text.splitlines()
    return "\n".join(parts[-lines:])
