import logging
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.organization import Organization
from app.models.query_log import QueryLog
from app.services.embedder import EmbedderService
from app.services.qdrant_client import QdrantService
from app.services.retriever import RetrievalService
from app.services.generator import GeneratorService
from app.services.langfuse_service import LangfuseService
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orgs", tags=["query"])

# Module-level service instances
_embedder = EmbedderService()
_qdrant = QdrantService()
_retrieval = RetrievalService(_embedder, _qdrant)
_generator = GeneratorService()
_langfuse = LangfuseService()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    doc_id: str
    chunk_id: str
    filename: str | None = None
    excerpt: str
    score: float
    citation_number: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    confidence: float
    low_confidence: bool
    trace_id: str
    latency_ms: int


@router.post("/{org_id}/query", response_model=QueryResponse)
async def query_documents(
    org_id: str,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    # Verify org exists
    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = org_result.scalar_one_or_none()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    start_time = time.monotonic()
    settings = get_settings()

    # Create Langfuse trace (no-op if not configured)
    trace_id = _langfuse.create_trace(
        name="rag-query",
        org_id=org_id,
        query=body.query,
    )

    # Step 1: Retrieve relevant chunks
    retrieval_start = time.monotonic()
    results, confidence, low_confidence = await _retrieval.retrieve(
        org_id=org_id,
        query=body.query,
        top_k=body.top_k,
    )
    retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

    _langfuse.log_retrieval(
        trace_id=trace_id,
        query=body.query,
        results=results,
        latency_ms=retrieval_ms,
    )

    # Step 2: Generate answer
    gen_start = time.monotonic()
    generation = await _generator.generate(
        query=body.query,
        context_chunks=results,
        confidence=confidence,
        low_confidence=low_confidence,
    )
    gen_ms = int((time.monotonic() - gen_start) * 1000)

    _langfuse.log_generation(
        trace_id=trace_id,
        prompt=body.query,
        response=generation.answer,
        tokens={
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
        },
        latency_ms=gen_ms,
        confidence=confidence,
    )

    total_ms = int((time.monotonic() - start_time) * 1000)

    # Build doc_id -> filename map for source enrichment
    doc_ids = {s.doc_id for s in generation.sources}
    filename_map: dict[str, str] = {}
    if doc_ids:
        from app.models.document import Document
        doc_result = await db.execute(
            select(Document).where(
                Document.id.in_(doc_ids),
                Document.org_id == org_id,  # enforce org isolation
            )
        )
        for doc in doc_result.scalars().all():
            filename_map[doc.id] = doc.filename

    # Build source response (only cited chunks)
    sources = [
        SourceResponse(
            doc_id=s.doc_id,
            chunk_id=s.chunk_id,
            filename=filename_map.get(s.doc_id),
            excerpt=s.excerpt,
            score=round(s.score, 4),
            citation_number=s.citation_number,
        )
        for s in generation.sources
    ]

    # Save query log
    log = QueryLog(
        id=str(uuid.uuid4()),
        org_id=org_id,
        query_text=body.query,
        answer=generation.answer,
        sources=[
            {
                "doc_id": s.doc_id,
                "chunk_id": s.chunk_id,
                "score": s.score,
                "excerpt": s.excerpt,
            }
            for s in generation.sources
        ],
        confidence_score=confidence,
        llm_model=settings.llm_model,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        latency_ms=total_ms,
        langfuse_trace_id=trace_id,
    )
    db.add(log)
    await db.commit()

    return QueryResponse(
        answer=generation.answer,
        sources=sources,
        confidence=round(confidence, 4),
        low_confidence=low_confidence,
        trace_id=trace_id,
        latency_ms=total_ms,
    )
