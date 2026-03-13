from pydantic import BaseModel

class ProductCard(BaseModel):
    id: int
    name: str
    price: float
    image_url: str = ""
    link: str = ""

class ChatRequest(BaseModel):
    message: str
    id_customer: int | None = None
    thread_id: str | None = None

class ChatResponse(BaseModel):
    reply: str
    thread_id: str | None = None
    escalated: bool = False
    products: list[ProductCard] = []

class ChatHistoryRequest(BaseModel):
    thread_id: str
    id_customer: int | None = None

class ChatMessageOut(BaseModel):
    role: str
    content: str
    products: list[ProductCard] = []

class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessageOut]

class ClearChatRequest(BaseModel):
    thread_id: str
