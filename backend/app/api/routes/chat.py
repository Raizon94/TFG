from fastapi import APIRouter
import json as _json
from starlette.responses import StreamingResponse

from app.schemas.chat import (
    ChatRequest, ChatResponse, ProductCard, ChatHistoryRequest, 
    ChatHistoryResponse, ChatMessageOut, ClearChatRequest
)
from app.agents.customer import agent as customer_agent
from app.agents.admin import agent as admin_agent

router = APIRouter()

@router.post("/customer", response_model=ChatResponse)
def chat_customer(req: ChatRequest):
    result = customer_agent.chat(
        message=req.message,
        id_customer=req.id_customer,
        thread_id=req.thread_id,
    )
    return ChatResponse(
        reply=result["reply"],
        thread_id=result["thread_id"],
        escalated=result["escalated"],
        products=[ProductCard(**p) for p in result.get("products", [])],
    )

@router.post("/customer/stream")
def chat_customer_stream(req: ChatRequest):
    def event_generator():
        for event in customer_agent.chat_stream(
            message=req.message,
            id_customer=req.id_customer,
            thread_id=req.thread_id,
        ):
            yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/customer/history", response_model=ChatHistoryResponse)
def chat_customer_history(req: ChatHistoryRequest):
    result = customer_agent.get_chat_history(thread_id=req.thread_id)
    return ChatHistoryResponse(
        thread_id=result["thread_id"],
        messages=[
            ChatMessageOut(
                role=m.get("role", "assistant"),
                content=m.get("content", ""),
                products=[ProductCard(**p) for p in m.get("products", [])],
            )
            for m in result.get("messages", []) if m.get("content")
        ],
    )

@router.post("/customer/clear")
def chat_customer_clear(req: ClearChatRequest):
    ok = customer_agent.clear_chat_history(thread_id=req.thread_id)
    return {"ok": ok}

@router.post("/admin", response_model=ChatResponse)
def chat_admin(req: ChatRequest):
    result = admin_agent.chat(
        message=req.message,
        thread_id=req.thread_id,
    )
    return ChatResponse(
        reply=result["reply"],
        thread_id=result["thread_id"],
    )
