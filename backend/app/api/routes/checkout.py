from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from decimal import Decimal
import random
import string

from app.db.database import get_db, PsCart, PsOrders, PsOrderDetail, PsOrderHistory, PsOrderPayment, PsCustomer
from app.schemas.order import CheckoutRequest, CheckoutResponse
from app.services import ml_engine

router = APIRouter()

_CARRIER_ID = 81
_CURRENCY_ID = 1
_LANG_ID = 1
_TAX_RATE = Decimal("21.000")
_ORDER_STATE_PAID = 2

def _generate_reference() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=9))

@router.post("", response_model=CheckoutResponse)
def checkout(req: CheckoutRequest, db: Session = Depends(get_db)):
    if not req.items:
        raise HTTPException(status_code=400, detail="El carrito está vacío.")

    customer = db.query(PsCustomer).filter(PsCustomer.id_customer == req.id_customer).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Cliente no encontrado.")
    secure_key = customer.secure_key

    now = datetime.now()
    reference = _generate_reference()

    total_products_excl = Decimal(0)
    for item in req.items:
        total_products_excl += Decimal(str(item.unit_price)) * item.quantity

    total_products_incl = (total_products_excl * (1 + _TAX_RATE / 100)).quantize(Decimal("0.000001"))
    shipping_excl = Decimal("4.950000")
    shipping_incl = Decimal("5.990000")
    total_paid_excl = total_products_excl + shipping_excl
    total_paid_incl = total_products_incl + shipping_incl
    total_paid_real = total_paid_incl

    try:
        cart = PsCart(
            id_carrier=_CARRIER_ID, delivery_option="", id_lang=_LANG_ID,
            id_address_delivery=req.id_address, id_address_invoice=req.id_address,
            id_currency=_CURRENCY_ID, id_customer=req.id_customer, id_guest=0,
            secure_key=secure_key, date_add=now, date_upd=now,
        )
        db.add(cart)
        db.flush()

        order = PsOrders(
            reference=reference, id_carrier=_CARRIER_ID, id_lang=_LANG_ID, id_customer=req.id_customer,
            id_cart=cart.id_cart, id_currency=_CURRENCY_ID, id_address_delivery=req.id_address,
            id_address_invoice=req.id_address, current_state=_ORDER_STATE_PAID, secure_key=secure_key,
            payment="Pago simulado (TFG Demo)", module="tfg_demo", valid=1, total_discounts=0,
            total_discounts_tax_incl=0, total_discounts_tax_excl=0, total_paid=total_paid_incl,
            total_paid_tax_incl=total_paid_incl, total_paid_tax_excl=total_paid_excl,
            total_paid_real=total_paid_real, total_products=total_products_excl,
            total_products_wt=total_products_incl, total_shipping=shipping_incl,
            total_shipping_tax_incl=shipping_incl, total_shipping_tax_excl=shipping_excl,
            carrier_tax_rate=_TAX_RATE, total_wrapping=0, total_wrapping_tax_incl=0,
            total_wrapping_tax_excl=0, invoice_date=now, delivery_date=now, date_add=now, date_upd=now,
        )
        db.add(order)
        db.flush()

        for item in req.items:
            unit_price_excl = Decimal(str(item.unit_price))
            unit_price_incl = (unit_price_excl * (1 + _TAX_RATE / 100)).quantize(Decimal("0.000001"))
            total_price_excl = unit_price_excl * item.quantity
            total_price_incl = unit_price_incl * item.quantity

            detail = PsOrderDetail(
                id_order=order.id_order, product_id=item.product_id, product_name=item.product_name,
                product_quantity=item.quantity, product_price=unit_price_excl,
                tax_rate=_TAX_RATE, total_price_tax_incl=total_price_incl, total_price_tax_excl=total_price_excl,
                unit_price_tax_incl=unit_price_incl, unit_price_tax_excl=unit_price_excl,
            )
            db.add(detail)

        history = PsOrderHistory(
            id_order=order.id_order, id_order_state=_ORDER_STATE_PAID, date_add=now
        )
        db.add(history)

        payment = PsOrderPayment(
            order_reference=reference, amount=total_paid_real, payment_method="Simulador Stripe", date_add=now
        )
        db.add(payment)

        db.commit()

        should_retrain, _ = ml_engine.check_retrain_needed(db)
        if should_retrain:
            import app.main as b_main
            ml_engine.trigger_retrain_async(reload_callback=b_main._reload_models)

        return CheckoutResponse(
            success=True, order_id=order.id_order, reference=reference, total_paid=float(total_paid_real),
            message="Pedido creado con éxito en PrestaShop."
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
