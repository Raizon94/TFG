from pydantic import BaseModel

class CartItem(BaseModel):
    product_id: int
    product_name: str
    quantity: int = 1
    unit_price: float

class CheckoutRequest(BaseModel):
    items: list[CartItem]
    id_customer: int
    id_address: int

class CheckoutResponse(BaseModel):
    success: bool
    order_id: int
    reference: str
    total_paid: float
    message: str
