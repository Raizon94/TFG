from fastapi import APIRouter
from app.api.routes import catalog, chat, admin, checkout, customers

api_router = APIRouter()
api_router.include_router(catalog.router, tags=["catalog"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(checkout.router, prefix="/checkout", tags=["checkout"])
api_router.include_router(customers.router, prefix="/customers", tags=["customers"])
