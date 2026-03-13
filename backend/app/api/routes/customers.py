from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.db.database import get_db, PsCustomer, PsAddress, PsOrders
from app.schemas.admin import TopCustomerOut

router = APIRouter()

@router.get("/top", response_model=list[TopCustomerOut])
def get_top_customers(db: Session = Depends(get_db)):
    spent_subq = (
        db.query(
            PsOrders.id_customer,
            func.sum(PsOrders.total_paid_real).label("total_spent")
        )
        .filter(PsOrders.valid == 1)
        .group_by(PsOrders.id_customer)
        .subquery()
    )

    rows = (
        db.query(
            PsCustomer.id_customer,
            PsCustomer.firstname,
            PsCustomer.lastname,
            PsCustomer.email,
            func.min(PsAddress.id_address).label("id_address"),
            spent_subq.c.total_spent
        )
        .join(spent_subq, PsCustomer.id_customer == spent_subq.c.id_customer)
        .join(PsAddress, PsCustomer.id_customer == PsAddress.id_customer)
        .filter(
            PsCustomer.active == 1,
            PsCustomer.deleted == 0,
            PsAddress.active == 1,
            PsAddress.deleted == 0
        )
        .group_by(
            PsCustomer.id_customer,
            PsCustomer.firstname,
            PsCustomer.lastname,
            PsCustomer.email,
            spent_subq.c.total_spent
        )
        .order_by(desc(spent_subq.c.total_spent))
        .limit(10)
        .all()
    )

    return [
        TopCustomerOut(
            id_customer=r.id_customer,
            name=f"{r.firstname} {r.lastname}",
            email=r.email,
            id_address=r.id_address,
            total_spent=float(r.total_spent or 0)
        )
        for r in rows
    ]
