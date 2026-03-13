from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.db.database import get_db, PsProduct, PsProductLang, PsImage, PsOrderDetail
from app.api.utils import build_prestashop_image_url
from app.schemas.product import ProductOut, RecommendationOut
from app.services import ml_engine
import app.main as b_main

router = APIRouter()

def _rows_to_products(rows: list) -> list[ProductOut]:
    products = []
    for prod, lang, img in rows:
        products.append(
            ProductOut(
                id=prod.id_product,
                name=lang.name,
                price=float(prod.price),
                image_url=build_prestashop_image_url(img.id_image),
            )
        )
    return products

@router.get("/products", response_model=list[ProductOut])
def get_products(db: Session = Depends(get_db)):
    newest_rows = (
        db.query(PsProduct, PsProductLang, PsImage)
        .join(PsProductLang, PsProduct.id_product == PsProductLang.id_product)
        .join(PsImage, PsProduct.id_product == PsImage.id_product)
        .filter(
            PsProductLang.id_lang == 3,
            PsImage.cover == 1,
            PsProduct.active == 1,
            PsProduct.price > 0,
        )
        .order_by(PsProduct.date_add.desc())
        .limit(4)
        .all()
    )

    newest_ids = {row[0].id_product for row in newest_rows}

    popular_sub = (
        db.query(
            PsOrderDetail.product_id,
            func.sum(PsOrderDetail.product_quantity).label("total_sold"),
        )
        .group_by(PsOrderDetail.product_id)
        .subquery()
    )

    popular_query = (
        db.query(PsProduct, PsProductLang, PsImage)
        .join(PsProductLang, PsProduct.id_product == PsProductLang.id_product)
        .join(PsImage, PsProduct.id_product == PsImage.id_product)
        .join(popular_sub, PsProduct.id_product == popular_sub.c.product_id)
        .filter(
            PsProductLang.id_lang == 3,
            PsImage.cover == 1,
            PsProduct.active == 1,
            PsProduct.price > 0,
        )
    )

    if newest_ids:
        popular_query = popular_query.filter(~PsProduct.id_product.in_(newest_ids))

    popular_rows = (
        popular_query
        .order_by(desc(popular_sub.c.total_sold))
        .limit(8)
        .all()
    )

    return _rows_to_products(list(newest_rows) + list(popular_rows))


@router.get("/recommendations/{user_id}", response_model=RecommendationOut)
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    recommended_ids = []
    
    if b_main.ml_what is not None:
        recommended_ids = ml_engine.predict_what(user_id, db, b_main.ml_what)

    if not recommended_ids and b_main.ml_fallback:
        recommended_ids = b_main.ml_fallback[:10]

    if not recommended_ids:
        subq = (
            db.query(PsProduct.id_product)
            .join(PsImage, PsProduct.id_product == PsImage.id_product)
            .filter(PsProduct.active == 1, PsImage.cover == 1)
            .all()
        )
        all_ids = [r[0] for r in subq]
        import random
        rng = random.Random(user_id)
        recommended_ids = rng.sample(all_ids, min(3, len(all_ids)))

    rows = (
        db.query(PsProduct, PsProductLang, PsImage)
        .join(PsProductLang, PsProduct.id_product == PsProductLang.id_product)
        .join(PsImage, PsProduct.id_product == PsImage.id_product)
        .filter(
            PsProduct.id_product.in_(recommended_ids),
            PsProductLang.id_lang == 1,
            PsImage.cover == 1,
            PsProduct.active == 1,
        )
        .all()
    )
    all_products = _rows_to_products(rows)

    rank = {pid: i for i, pid in enumerate(recommended_ids)}
    all_products.sort(key=lambda p: rank.get(p.id, 9999))
    products = all_products[:3]

    if len(products) < 3:
        existing_ids = {p.id for p in products}
        filler = (
            db.query(PsProduct, PsProductLang, PsImage)
            .join(PsProductLang, PsProduct.id_product == PsProductLang.id_product)
            .join(PsImage, PsProduct.id_product == PsImage.id_product)
            .filter(
                PsProductLang.id_lang == 1,
                PsImage.cover == 1,
                PsProduct.active == 1,
                ~PsProduct.id_product.in_(existing_ids),
            )
            .limit(3 - len(products))
            .all()
        )
        products.extend(_rows_to_products(filler))

    return RecommendationOut(recommendations=products)
