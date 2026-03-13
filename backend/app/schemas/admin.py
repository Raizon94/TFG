from pydantic import BaseModel

class AdminStats(BaseModel):
    total_pedidos: int
    ingresos_mes: float
    clientes_activos: int
    productos_activos: int
    tasa_conversion: float
    ticket_medio: float

class TopCustomerOut(BaseModel):
    id_customer: int
    name: str
    email: str
    id_address: int
    total_spent: float
