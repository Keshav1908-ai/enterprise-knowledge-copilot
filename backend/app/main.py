"""FastAPI application factory."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, system
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.database.session import init_db, session_scope
from app.monitoring.middleware import RequestMetricsMiddleware
from app.services.document_service import DocumentService

logger = get_logger(__name__)


def _resume_interrupted_ingestion() -> None:
    settings = get_settings()
    service = DocumentService(settings)
    with session_scope() as db:
        ids = service.recover_interrupted(db)
    if ids:
        logger.info("Resuming %d interrupted ingestion job(s)", len(ids))
        for doc_id in ids:
            service.process(doc_id)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_directories()
    init_db()
    logger.info(
        "Started %s v%s | embeddings=%s:%s | llm=%s | retrieval=%s top_k=%d",
        settings.app_name, settings.app_version, settings.embedding_provider, settings.embedding_model,
        settings.llm_provider, settings.retrieval_mode, settings.top_k,
    )
    if settings.environment != "test":
        threading.Thread(target=_resume_interrupted_ingestion, daemon=True).start()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "RAG-based assistant for enterprise documents: ingestion, semantic/hybrid retrieval, "
            "grounded answers with verifiable citations, conversation memory, evaluation and monitoring."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestMetricsMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )
    register_exception_handlers(app)
    app.include_router(system.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    return app


app = create_app()
