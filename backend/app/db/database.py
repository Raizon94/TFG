"""
backend/database.py — Conexión y modelos SQLAlchemy
===================================================================
Mapea las tablas principales de PrestaShop que necesita la API:
  • ps_product        — catálogo (precio, estado)
  • ps_product_lang   — traducciones (nombre, descripción)
  • ps_image          — imágenes (cover)
  • ps_cart           — carrito de compra
  • ps_orders         — pedidos
  • ps_order_detail   — líneas de pedido
  • ps_order_history  — historial de estados
  • ps_order_payment  — pagos registrados
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    SmallInteger,
    String,
    Text,
    DECIMAL,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
)
import logging

# ─── Configuración de conexión ──────────────────────────────────────
import os

DATABASE_URL: str = os.getenv("DATABASE_URL", "mysql+pymysql://root:root1234@localhost:3306/tfg_bd")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()
logger = logging.getLogger("uvicorn.error")


def ensure_ai_tables() -> None:
    """Crea tablas auxiliares de IA si no existen.

    Algunos entornos Docker arrancan con dumps antiguos que no incluyen
    ps_ai_boost y ps_ai_product_attr. Estas tablas son opcionales para
    el core de PrestaShop pero necesarias para rutas de IA.
    """
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS ps_ai_boost (
            id_product INT NOT NULL,
            boost_factor DECIMAL(4,2) NOT NULL DEFAULT 1.50,
            reason VARCHAR(255) NULL,
            date_add DATETIME NOT NULL,
            date_upd DATETIME NOT NULL,
            PRIMARY KEY (id_product)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS ps_ai_product_attr (
            id_product INT NOT NULL,
            stone VARCHAR(80) NOT NULL DEFAULT '',
            type VARCHAR(80) NOT NULL DEFAULT '',
            updated_at DATETIME NULL,
            PRIMARY KEY (id_product),
            KEY idx_ai_attr_stone (stone),
            KEY idx_ai_attr_type (type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
    ]

    try:
        with engine.begin() as conn:
            for ddl in ddl_statements:
                conn.execute(text(ddl))
        logger.info("Tablas IA verificadas: ps_ai_boost, ps_ai_product_attr")
    except Exception as exc:
        logger.warning("No se pudieron verificar tablas IA: %s", exc)


# ─── Dependency de FastAPI ──────────────────────────────────────────
def get_db() -> Session:  # type: ignore[misc]
    """Genera una sesión de BD por request y la cierra al terminar."""
    db = SessionLocal()
    try:
        yield db  # type: ignore[misc]
    finally:
        db.close()


# ─── Modelos (solo lectura, reflejan las tablas existentes) ─────────

class PsProduct(Base):
    """Tabla ps_product — datos maestros del producto."""

    __tablename__ = "ps_product"

    id_product: int = Column(Integer, primary_key=True, autoincrement=True)
    price = Column(DECIMAL(20, 6), nullable=False, default=0)
    active: int = Column(Integer, nullable=False, default=0)
    reference: str | None = Column(String(64))
    quantity: int = Column(Integer, nullable=False, default=0)
    date_add = Column(DateTime)


class PsProductLang(Base):
    """Tabla ps_product_lang — nombre y descripción por idioma."""

    __tablename__ = "ps_product_lang"

    id_product: int = Column(Integer, primary_key=True)
    id_shop: int = Column(Integer, primary_key=True)
    id_lang: int = Column(Integer, primary_key=True)
    name: str = Column(String(128), nullable=False)
    description_short: str | None = Column(Text)
    description: str | None = Column(Text)
    link_rewrite: str = Column(String(128), nullable=False)


class PsImage(Base):
    """Tabla ps_image — relación producto ↔ imagen (cover = portada)."""

    __tablename__ = "ps_image"

    id_image: int = Column(Integer, primary_key=True, autoincrement=True)
    id_product: int = Column(Integer, nullable=False, index=True)
    position: int = Column(SmallInteger, nullable=False, default=0)
    cover: int | None = Column(Integer)


# ─── Modelos de clientes ────────────────────────────────────────────

class PsCustomer(Base):
    """Tabla ps_customer — clientes de la tienda."""

    __tablename__ = "ps_customer"

    id_customer: int = Column(Integer, primary_key=True, autoincrement=True)
    firstname: str = Column(String(255), nullable=False)
    lastname: str = Column(String(255), nullable=False)
    email: str = Column(String(255), nullable=False)
    secure_key: str = Column(String(32), nullable=False, default="-1")
    active: int = Column(Integer, nullable=False, default=0)
    deleted: int = Column(Integer, nullable=False, default=0)


class PsAddress(Base):
    """Tabla ps_address — direcciones de los clientes."""

    __tablename__ = "ps_address"

    id_address: int = Column(Integer, primary_key=True, autoincrement=True)
    id_customer: int = Column(Integer, nullable=False, default=0)
    alias: str = Column(String(32), nullable=False)
    firstname: str = Column(String(255), nullable=False)
    lastname: str = Column(String(255), nullable=False)
    address1: str = Column(String(128), nullable=False)
    city: str = Column(String(64), nullable=False)
    active: int = Column(Integer, nullable=False, default=1)
    deleted: int = Column(Integer, nullable=False, default=0)


# ─── Modelos de pedidos / checkout ──────────────────────────────────

class PsCart(Base):
    """Tabla ps_cart — carrito de compra."""

    __tablename__ = "ps_cart"

    id_cart: int = Column(Integer, primary_key=True, autoincrement=True)
    id_shop_group: int = Column(Integer, nullable=False, default=1)
    id_shop: int = Column(Integer, nullable=False, default=1)
    id_carrier: int = Column(Integer, nullable=False, default=0)
    delivery_option: str = Column(Text, nullable=False, default="")
    id_lang: int = Column(Integer, nullable=False, default=1)
    id_address_delivery: int = Column(Integer, nullable=False, default=0)
    id_address_invoice: int = Column(Integer, nullable=False, default=0)
    id_currency: int = Column(Integer, nullable=False, default=1)
    id_customer: int = Column(Integer, nullable=False, default=0)
    id_guest: int = Column(Integer, nullable=False, default=0)
    secure_key: str = Column(String(32), nullable=False, default="-1")
    recyclable: int = Column(Integer, nullable=False, default=1)
    gift: int = Column(Integer, nullable=False, default=0)
    gift_message: str | None = Column(Text)
    mobile_theme: int = Column(Integer, nullable=False, default=0)
    allow_seperated_package: int = Column(Integer, nullable=False, default=0)
    date_add = Column(DateTime, nullable=False)
    date_upd = Column(DateTime, nullable=False)


class PsOrders(Base):
    """Tabla ps_orders — pedidos."""

    __tablename__ = "ps_orders"

    id_order: int = Column(Integer, primary_key=True, autoincrement=True)
    reference: str | None = Column(String(9))
    id_shop_group: int = Column(Integer, nullable=False, default=1)
    id_shop: int = Column(Integer, nullable=False, default=1)
    id_carrier: int = Column(Integer, nullable=False)
    id_lang: int = Column(Integer, nullable=False, default=1)
    id_customer: int = Column(Integer, nullable=False)
    id_cart: int = Column(Integer, nullable=False)
    id_currency: int = Column(Integer, nullable=False, default=1)
    id_address_delivery: int = Column(Integer, nullable=False)
    id_address_invoice: int = Column(Integer, nullable=False)
    current_state: int = Column(Integer, nullable=False, default=0)
    secure_key: str = Column(String(32), nullable=False, default="-1")
    payment: str = Column(String(255), nullable=False)
    conversion_rate = Column(DECIMAL(13, 6), nullable=False, default=1)
    module: str | None = Column(String(255))
    recyclable: int = Column(Integer, nullable=False, default=0)
    gift: int = Column(Integer, nullable=False, default=0)
    gift_message: str | None = Column(Text)
    mobile_theme: int = Column(Integer, nullable=False, default=0)
    shipping_number: str | None = Column(String(64))
    total_discounts = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_discounts_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_discounts_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_paid = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_paid_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_paid_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_paid_real = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_products = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_products_wt = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_shipping = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_shipping_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_shipping_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    carrier_tax_rate = Column(DECIMAL(10, 3), nullable=False, default=0)
    total_wrapping = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_wrapping_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_wrapping_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    round_mode: int = Column(Integer, nullable=False, default=2)
    round_type: int = Column(Integer, nullable=False, default=1)
    invoice_number: int = Column(Integer, nullable=False, default=0)
    delivery_number: int = Column(Integer, nullable=False, default=0)
    invoice_date = Column(DateTime, nullable=False)
    delivery_date = Column(DateTime, nullable=False)
    valid: int = Column(Integer, nullable=False, default=0)
    date_add = Column(DateTime, nullable=False)
    date_upd = Column(DateTime, nullable=False)


class PsOrderDetail(Base):
    """Tabla ps_order_detail — líneas de un pedido."""

    __tablename__ = "ps_order_detail"

    id_order_detail: int = Column(Integer, primary_key=True, autoincrement=True)
    id_order: int = Column(Integer, nullable=False)
    id_order_invoice: int | None = Column(Integer, default=0)
    id_warehouse: int | None = Column(Integer, default=0)
    id_shop: int = Column(Integer, nullable=False, default=1)
    product_id: int = Column(Integer, nullable=False)
    product_attribute_id: int | None = Column(Integer, default=0)
    id_customization: int | None = Column(Integer, default=0)
    product_name: str = Column(String(255), nullable=False)
    product_quantity: int = Column(Integer, nullable=False, default=0)
    product_quantity_in_stock: int = Column(Integer, nullable=False, default=0)
    product_quantity_refunded: int = Column(Integer, nullable=False, default=0)
    product_quantity_return: int = Column(Integer, nullable=False, default=0)
    product_quantity_reinjected: int = Column(Integer, nullable=False, default=0)
    product_price = Column(DECIMAL(20, 6), nullable=False, default=0)
    reduction_percent = Column(DECIMAL(10, 2), nullable=False, default=0)
    reduction_amount = Column(DECIMAL(20, 6), nullable=False, default=0)
    reduction_amount_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    reduction_amount_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    group_reduction = Column(DECIMAL(10, 2), nullable=False, default=0)
    product_quantity_discount = Column(DECIMAL(20, 6), nullable=False, default=0)
    product_ean13: str | None = Column(String(13))
    product_isbn: str | None = Column(String(32))
    product_upc: str | None = Column(String(12))
    product_reference: str | None = Column(String(64))
    product_supplier_reference: str | None = Column(String(64))
    product_weight = Column(DECIMAL(20, 6), nullable=False, default=0)
    id_tax_rules_group: int | None = Column(Integer, default=0)
    tax_computation_method: int = Column(Integer, nullable=False, default=0)
    tax_name: str = Column(String(16), nullable=False, default="")
    tax_rate = Column(DECIMAL(10, 3), nullable=False, default=0)
    ecotax = Column(DECIMAL(21, 6), nullable=False, default=0)
    ecotax_tax_rate = Column(DECIMAL(5, 3), nullable=False, default=0)
    discount_quantity_applied: int = Column(Integer, nullable=False, default=0)
    download_hash: str | None = Column(String(255))
    download_nb: int | None = Column(Integer, default=0)
    download_deadline = Column(DateTime, nullable=True)
    total_price_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_price_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    unit_price_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    unit_price_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_shipping_price_tax_incl = Column(DECIMAL(20, 6), nullable=False, default=0)
    total_shipping_price_tax_excl = Column(DECIMAL(20, 6), nullable=False, default=0)
    purchase_supplier_price = Column(DECIMAL(20, 6), nullable=False, default=0)
    original_product_price = Column(DECIMAL(20, 6), nullable=False, default=0)
    original_wholesale_price = Column(DECIMAL(20, 6), nullable=False, default=0)


class PsOrderHistory(Base):
    """Tabla ps_order_history — historial de estados de un pedido."""

    __tablename__ = "ps_order_history"

    id_order_history: int = Column(Integer, primary_key=True, autoincrement=True)
    id_employee: int = Column(Integer, nullable=False, default=0)
    id_order: int = Column(Integer, nullable=False)
    id_order_state: int = Column(Integer, nullable=False)
    date_add = Column(DateTime, nullable=False)


class PsOrderStateLang(Base):
    """Tabla ps_order_state_lang — nombre legible del estado de pedido."""

    __tablename__ = "ps_order_state_lang"

    id_order_state: int = Column(Integer, primary_key=True)
    id_lang: int = Column(Integer, primary_key=True)
    name: str = Column(String(64), nullable=False)
    template: str = Column(String(64), nullable=False, default="")


class PsOrderPayment(Base):
    """Tabla ps_order_payment — pagos asociados a un pedido."""

    __tablename__ = "ps_order_payment"

    id_order_payment: int = Column(Integer, primary_key=True, autoincrement=True)
    order_reference: str | None = Column(String(9))
    id_currency: int = Column(Integer, nullable=False, default=1)
    amount = Column(DECIMAL(10, 2), nullable=False)
    payment_method: str = Column(String(255), nullable=False)
    conversion_rate = Column(DECIMAL(13, 6), nullable=False, default=1)
    transaction_id: str | None = Column(String(254))
    card_number: str | None = Column(String(254))
    card_brand: str | None = Column(String(254))
    card_expiration: str | None = Column(String(7))
    card_holder: str | None = Column(String(254))
    date_add = Column(DateTime, nullable=False)


class PsAiBoost(Base):
    """Tabla ps_ai_boost — productos potenciados para el asistente IA."""

    __tablename__ = "ps_ai_boost"

    id_product: int = Column(Integer, primary_key=True)
    boost_factor: float = Column(DECIMAL(4, 2), nullable=False, default=1.5)
    reason: str | None = Column(String(255))
    date_add = Column(DateTime, nullable=False)
    date_upd = Column(DateTime, nullable=False)


class PsAiProductAttr(Base):
    """Tabla ps_ai_product_attr — tipo y piedra extraídos por IA."""

    __tablename__ = "ps_ai_product_attr"

    id_product: int = Column(Integer, primary_key=True)
    stone: str = Column(String(80), nullable=False, default="")
    type: str = Column(String(80), nullable=False, default="")
    updated_at = Column(DateTime, nullable=True)
