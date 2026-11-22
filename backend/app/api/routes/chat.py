from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.database.models import Conversation, Message
from app.database.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ConversationDetail, ConversationSummary, MessageOut
from app.schemas.documents import ActionResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Ask a question (RAG with citations)")
async def chat(req: ChatRequest, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> ChatResponse:
    """Runs the full RAG pipeline. Omit ``conversation_id`` to start a new conversation."""
    service = ChatService(settings)
    result = await run_in_threadpool(
        service.ask,
        db,
        req.question,
        conversation_id=req.conversation_id,
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        retrieval_mode=req.retrieval_mode,
        reranker=req.reranker,
        document_ids=req.document_ids,
    )
    return ChatResponse(**result.as_dict())


@router.get("/conversations", response_model=list[ConversationSummary], summary="List conversations")
def list_conversations(db: Session = Depends(get_db), limit: int = 50) -> list[ConversationSummary]:
    limit = max(1, min(limit, 200))
    counts = dict(db.execute(select(Message.conversation_id, func.count()).group_by(Message.conversation_id)).all())
    convs = db.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit))
    out = []
    for c in convs:
        last = db.scalar(
            select(Message.content).where(Message.conversation_id == c.id).order_by(Message.created_at.desc()).limit(1)
        )
        out.append(ConversationSummary(
            id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at,
            message_count=counts.get(c.id, 0), last_message_preview=(last or "")[:140] or None,
        ))
    return out


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail, summary="Conversation history")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ConversationDetail:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise NotFoundError(f"Conversation '{conversation_id}' not found")
    return ConversationDetail(
        id=conv.id, title=conv.title, created_at=conv.created_at, updated_at=conv.updated_at,
        messages=[MessageOut.model_validate(m) for m in conv.messages],
    )


@router.delete("/conversations/{conversation_id}", response_model=ActionResponse, summary="Delete a conversation")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ActionResponse:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise NotFoundError(f"Conversation '{conversation_id}' not found")
    db.delete(conv)
    db.commit()
    return ActionResponse(id=conversation_id, status="deleted", message="Conversation deleted")
