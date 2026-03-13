from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime

from app.db.database import get_db, PsOrders, PsCustomer, PsAddress, PsProduct, PsAiBoost, PsProductLang, PsImage
from app.schemas.admin import AdminStats, TopCustomerOut
from app.agents.admin import agent as admin_agent
from app.services import ml_engine
import app.main as b_main

router = APIRouter()

@router.post("/analyze-product-image")
async def analyze_product_image(
    image: UploadFile = File(...),
    message: str = Form(""),
    thread_id: str | None = Form(None),
):
    import base64
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    mime = image.content_type or "image/jpeg"
    if mime not in allowed:
        raise HTTPException(400, f"Tipo de imagen no soportado: {mime}. Usa JPEG, PNG o WebP.")

    raw = await image.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "La imagen es demasiado grande (máx 20 MB).")

    image_b64 = base64.b64encode(raw).decode("utf-8")

    from app.agents.admin.tools import stage_product_image
    tid = thread_id or ""
    if not tid:
        import uuid as _uuid
        tid = str(_uuid.uuid4())
    stage_product_image(tid, raw, mime)

    result = admin_agent.chat_with_image(
        image_b64=image_b64,
        mime_type=mime,
        message=message,
        thread_id=tid,
    )

    return {
        "reply": result["reply"],
        "thread_id": result["thread_id"],
        "vision_data": result.get("vision_data"),
    }

@router.get("/stats", response_model=AdminStats)
def admin_stats(db: Session = Depends(get_db)):
    now = datetime.now()
    first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    total_pedidos: int = db.query(func.count(PsOrders.id_order)).filter(PsOrders.valid == 1).scalar() or 0
    ingresos_mes: float = float(db.query(func.coalesce(func.sum(PsOrders.total_paid_real), 0)).filter(
        PsOrders.valid == 1, PsOrders.date_add >= first_of_month
    ).scalar() or 0)

    from sqlalchemy import distinct
    from datetime import timedelta
    since_1y = now - timedelta(days=365)
    
    clientes_activos: int = db.query(func.count(distinct(PsOrders.id_customer))).filter(
        PsOrders.valid == 1, PsOrders.date_add >= since_1y
    ).scalar() or 0

    productos_activos: int = db.query(func.count(PsProduct.id_product)).filter(PsProduct.active == 1).scalar() or 0

    ticket_medio: float = float(db.query(func.coalesce(func.avg(PsOrders.total_paid_real), 0)).filter(
        PsOrders.valid == 1
    ).scalar() or 0)

    clientes_mes: int = db.query(func.count(distinct(PsOrders.id_customer))).filter(
        PsOrders.valid == 1, PsOrders.date_add >= first_of_month
    ).scalar() or 0

    tasa_conversion: float = round((clientes_mes / clientes_activos) * 100, 2) if clientes_activos > 0 else 0.0

    return AdminStats(
        total_pedidos=total_pedidos,
        ingresos_mes=round(ingresos_mes, 2),
        clientes_activos=clientes_activos,
        productos_activos=productos_activos,
        tasa_conversion=tasa_conversion,
        ticket_medio=round(ticket_medio, 2),
    )

@router.get("/model-status")
def get_model_status(db: Session = Depends(get_db)):
    meta = ml_engine.get_training_metadata()
    should_retrain, retrain_reason = ml_engine.check_retrain_needed(db)
    
    from pathlib import Path
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    
    return {
        "status": "ok",
        "svd_loaded": b_main.ml_what is not None,
        "fallback_loaded": b_main.ml_fallback is not None,
        "is_retraining": ml_engine.is_retraining(),
        "should_retrain": should_retrain,
        "retrain_reason": retrain_reason,
        "last_trained_orders": meta.get("order_count") if meta else None,
        "last_trained_date": meta.get("timestamp") if meta else None,
        "svd_file_size_kb": round((models_dir / "svd_what.joblib").stat().st_size / 1024, 1) if (models_dir / "svd_what.joblib").exists() else 0,
    }

@router.post("/retrain")
def trigger_retrain():
    started = ml_engine.trigger_retrain_async(reload_callback=b_main._reload_models)
    if not started:
        raise HTTPException(400, "Ya hay un reentrenamiento en curso.")
    return {"status": "started", "message": "Reentrenamiento lanzado en background"}

@router.get("/retrain-log")
def get_retrain_log():
    return {"log": ml_engine.get_retrain_log(lines=300)}

@router.get("/boost")
def list_boosts(db: Session = Depends(get_db)):
    rows = (
        db.query(PsAiBoost, PsProductLang.name)
        .join(PsProductLang, PsAiBoost.id_product == PsProductLang.id_product)
        .filter(PsProductLang.id_lang == 3)
        .all()
    )
    return [
        {
            "id_product": row.PsAiBoost.id_product,
            "name": row.name,
            "boost_factor": float(row.PsAiBoost.boost_factor),
            "reason": row.PsAiBoost.reason,
            "date_add": row.PsAiBoost.date_add,
            "date_upd": row.PsAiBoost.date_upd,
        }
        for row in rows
    ]

from pydantic import BaseModel
class BoostCreate(BaseModel):
    id_product: int
    boost_factor: float
    reason: str | None = None

@router.post("/boost")
def set_boost(req: BoostCreate, db: Session = Depends(get_db)):
    if req.boost_factor < 1.0 or req.boost_factor > 5.0:
        raise HTTPException(400, "boost_factor debe estar entre 1.0 y 5.0")
    now = datetime.now()
    boost = db.query(PsAiBoost).filter(PsAiBoost.id_product == req.id_product).first()
    if boost:
        boost.boost_factor = req.boost_factor
        boost.reason = req.reason
        boost.date_upd = now
    else:
        boost = PsAiBoost(
            id_product=req.id_product,
            boost_factor=req.boost_factor,
            reason=req.reason,
            date_add=now,
            date_upd=now,
        )
        db.add(boost)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "message": f"Boost guardado para producto {req.id_product}"}

@router.delete("/boost/{id_product}")
def delete_boost(id_product: int, db: Session = Depends(get_db)):
    boost = db.query(PsAiBoost).filter(PsAiBoost.id_product == id_product).first()
    if not boost:
        raise HTTPException(status_code=404, detail="Boost no encontrado")
    db.delete(boost)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
    return {"status": "deleted", "id_product": id_product}

@router.get("/boost/search")
def search_boost_products(q: str, db: Session = Depends(get_db)):
    from app.api.utils import build_prestashop_image_url

    query = (q or "").strip()
    if not query:
        return []

    is_product_id = query.isdigit()
    if not is_product_id and len(query) < 2:
        return []

    filters = [
        PsProductLang.id_lang == 3,
        PsProduct.active == 1,
    ]
    if is_product_id:
        filters.append(PsProduct.id_product == int(query))
    else:
        filters.append(PsProduct.reference.ilike(f"%{query}%"))

    rows = (
        db.query(PsProduct, PsProductLang, PsImage)
        .join(PsProductLang, PsProduct.id_product == PsProductLang.id_product)
        .outerjoin(PsImage, (PsProduct.id_product == PsImage.id_product) & (PsImage.cover == 1))
        .filter(*filters)
        .limit(10)
        .all()
    )
    result = []
    for prod, lang, img in rows:
        result.append({
            "id_product": prod.id_product,
            "name": lang.name,
            "reference": prod.reference,
            "price": float(prod.price),
            "has_boost": db.query(PsAiBoost).filter(PsAiBoost.id_product == prod.id_product).count() > 0,
            "image_url": build_prestashop_image_url(img.id_image) if img else None,
        })
    return result
