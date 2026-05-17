import hashlib
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.document import Document, DocumentStatus
from app.models.organization import Organization
from app.services.chunker import ChunkerService
from app.services.embedder import EmbedderService
from app.services.qdrant_client import QdrantService
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orgs", tags=["documents"])

# Module-level service instances (shared across requests)
_chunker = ChunkerService()
_embedder = EmbedderService()
_qdrant = QdrantService()


class DocumentResponse(BaseModel):
    id: str
    org_id: str
    filename: str
    content_hash: str
    chunk_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


async def _process_document(
    doc_id: str,
    org_id: str,
    filename: str,
    content: bytes,
) -> None:
    """Background task: chunk → embed → upsert to Qdrant → update status."""
    from app.database import SessionLocal

    settings = get_settings()

    async with SessionLocal() as db:
        try:
            # Extract text
            if filename.lower().endswith(".pdf"):
                text = _chunker.extract_text_from_pdf(content)
            else:
                text = content.decode("utf-8", errors="replace")

            # Chunk
            chunks = _chunker.chunk_document(
                text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )

            if not chunks:
                raise ValueError("No chunks extracted from document")

            # Embed all chunks
            texts = [c.text for c in chunks]
            embeddings = await _embedder.embed_batch(texts)

            # Build (chunk_id, text, embedding) tuples
            chunk_tuples = [
                (f"{doc_id}_chunk_{c.index}", c.text, emb)
                for c, emb in zip(chunks, embeddings)
            ]

            # Upsert to Qdrant
            await _qdrant.ensure_collection(org_id)
            await _qdrant.upsert_chunks(org_id, doc_id, chunk_tuples)

            # Update document status
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = DocumentStatus.ready
                doc.chunk_count = len(chunks)
                await db.commit()

            logger.info("Document %s processed: %d chunks", doc_id, len(chunks))

        except Exception as e:
            logger.error("Failed to process document %s: %s", doc_id, e)
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = DocumentStatus.failed
                await db.commit()


@router.post("/{org_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    org_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    # Verify org exists
    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    if not org_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    # Validate file type
    filename = file.filename or "upload"
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and TXT files are supported",
        )

    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    # Dedup check — same hash in same org = duplicate
    existing = await db.execute(
        select(Document).where(Document.org_id == org_id, Document.content_hash == content_hash)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A document with identical content already exists in this organization (hash: {content_hash[:16]}...)",
        )

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        org_id=org_id,
        filename=filename,
        content_hash=content_hash,
        chunk_count=0,
        status=DocumentStatus.processing,
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Queue background processing
    background_tasks.add_task(_process_document, doc_id, org_id, filename, content)

    return DocumentResponse.model_validate(doc)


@router.get("/{org_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    org_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    if not org_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    result = await db.execute(
        select(Document).where(Document.org_id == org_id).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [DocumentResponse.model_validate(d) for d in docs]


@router.delete("/{org_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    org_id: str,
    doc_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.org_id == org_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Delete from Qdrant first
    try:
        await _qdrant.delete_document(org_id, doc_id)
    except Exception as e:
        logger.warning("Qdrant deletion failed for doc %s: %s — removing from DB anyway", doc_id, e)

    await db.delete(doc)
    await db.commit()
