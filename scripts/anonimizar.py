"""
Script de Anonimización RGPD — TFG e-commerce headless IA
=========================================================
Recorre las tablas con Información Personal Identificable (PII)
de la BD PrestaShop y la sobrescribe con datos ficticios generados
por Faker (locale es_ES).  Ejecutar UNA SOLA VEZ antes de arrancar
la aplicación.

Uso:
    python scripts/anonimizar.py
"""

from __future__ import annotations

import time
from typing import Any

from faker import Faker
from sqlalchemy import create_engine, text

# ─── Configuración ──────────────────────────────────────────────────
DB_URL: str = "mysql+pymysql://root:root1234@localhost:3306/tfg_bd"
FAKER_LOCALE: str = "es_ES"
BATCH_SIZE: int = 500  # filas por commit (evita bloqueos largos)

fake = Faker(FAKER_LOCALE)
Faker.seed(42)  # reproducibilidad

engine = create_engine(DB_URL, echo=False)


# ─── Utilidades ─────────────────────────────────────────────────────
def _progreso(tabla: str, procesadas: int, total: int) -> None:
    """Imprime el progreso de la anonimización."""
    pct = (procesadas / total * 100) if total else 0
    print(f"  [{tabla}] {procesadas}/{total}  ({pct:.1f}%)")


# ─── 1. ps_customer ────────────────────────────────────────────────
def anonimizar_customers() -> None:
    """Anonimiza nombres, emails, fechas de nacimiento y campos fiscales."""
    tabla = "ps_customer"

    with engine.begin() as conn:
        total: int = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()  # type: ignore[assignment]
        print(f"\n🔒 Anonimizando {tabla} ({total} filas)…")

        rows = conn.execute(text(f"SELECT id_customer FROM {tabla} ORDER BY id_customer"))
        ids: list[int] = [r[0] for r in rows]

        for i, cid in enumerate(ids, start=1):
            conn.execute(
                text(
                    f"""
                    UPDATE {tabla} SET
                        firstname       = :fn,
                        lastname        = :ln,
                        email           = :email,
                        birthday        = :bday,
                        company         = :company,
                        siret           = :siret,
                        ape             = :ape,
                        website         = :website,
                        passwd          = :passwd,
                        ip_registration_newsletter = '0.0.0.0',
                        note            = NULL,
                        secure_key      = :skey
                    WHERE id_customer = :cid
                    """
                ),
                {
                    "fn": fake.first_name(),
                    "ln": fake.last_name(),
                    "email": fake.unique.safe_email(),
                    "bday": fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
                    "company": fake.company() if fake.boolean(chance_of_getting_true=30) else "",
                    "siret": "",
                    "ape": "",
                    "website": "",
                    "passwd": fake.sha256(),
                    "skey": fake.md5()[:32],
                    "cid": cid,
                },
            )

            if i % BATCH_SIZE == 0 or i == total:
                _progreso(tabla, i, total)

    print(f"  ✅ {tabla} completado.\n")


# ─── 2. ps_address ─────────────────────────────────────────────────
def anonimizar_addresses() -> None:
    """Anonimiza direcciones, teléfonos, DNI y datos fiscales."""
    tabla = "ps_address"

    with engine.begin() as conn:
        total: int = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()  # type: ignore[assignment]
        print(f"🔒 Anonimizando {tabla} ({total} filas)…")

        rows = conn.execute(text(f"SELECT id_address FROM {tabla} ORDER BY id_address"))
        ids: list[int] = [r[0] for r in rows]

        for i, aid in enumerate(ids, start=1):
            conn.execute(
                text(
                    f"""
                    UPDATE {tabla} SET
                        firstname    = :fn,
                        lastname     = :ln,
                        address1     = :addr1,
                        address2     = :addr2,
                        city         = :city,
                        postcode     = :pc,
                        phone        = :phone,
                        phone_mobile = :mobile,
                        company      = :company,
                        vat_number   = :vat,
                        dni          = :dni,
                        other        = NULL
                    WHERE id_address = :aid
                    """
                ),
                {
                    "fn": fake.first_name(),
                    "ln": fake.last_name(),
                    "addr1": fake.street_address(),
                    "addr2": "" if fake.boolean(chance_of_getting_true=70) else fake.secondary_address(),
                    "city": fake.city(),
                    "pc": fake.postcode(),
                    "phone": fake.phone_number()[:32],
                    "mobile": fake.phone_number()[:32],
                    "company": fake.company() if fake.boolean(chance_of_getting_true=20) else "",
                    "vat": "",
                    "dni": "",
                    "aid": aid,
                },
            )

            if i % BATCH_SIZE == 0 or i == total:
                _progreso(tabla, i, total)

    print(f"  ✅ {tabla} completado.\n")


# ─── 3. ps_customer_message  (texto libre — hilos de contacto) ─────
def anonimizar_mensajes_cliente() -> None:
    """Sustituye el contenido de mensajes de clientes por Lorem Ipsum."""
    tabla = "ps_customer_message"

    with engine.begin() as conn:
        total: int = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()  # type: ignore[assignment]
        print(f"🔒 Anonimizando {tabla} ({total} filas)…")

        if total == 0:
            print("  ⏭ Sin filas, saltando.\n")
            return

        conn.execute(
            text(
                f"""
                UPDATE {tabla} SET
                    message    = :lorem,
                    ip_address = '0.0.0.0',
                    user_agent = 'Anon/1.0'
                """
            ),
            {"lorem": fake.paragraph(nb_sentences=3)},
        )
        _progreso(tabla, total, total)

    print(f"  ✅ {tabla} completado.\n")


# ─── 4. ps_message  (mensajes internos de pedidos) ─────────────────
def anonimizar_mensajes_pedido() -> None:
    """Sustituye el contenido de mensajes de pedidos por Lorem Ipsum."""
    tabla = "ps_message"

    with engine.begin() as conn:
        total: int = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()  # type: ignore[assignment]
        print(f"🔒 Anonimizando {tabla} ({total} filas)…")

        if total == 0:
            print("  ⏭ Sin filas, saltando.\n")
            return

        conn.execute(
            text(f"UPDATE {tabla} SET message = :lorem"),
            {"lorem": fake.paragraph(nb_sentences=2)},
        )
        _progreso(tabla, total, total)

    print(f"  ✅ {tabla} completado.\n")


# ─── Main ───────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("   ANONIMIZACIÓN RGPD — PrestaShop → TFG")
    print("=" * 60)

    t0 = time.time()

    anonimizar_customers()
    anonimizar_addresses()
    anonimizar_mensajes_cliente()
    anonimizar_mensajes_pedido()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"   ✅ ANONIMIZACIÓN COMPLETADA en {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
