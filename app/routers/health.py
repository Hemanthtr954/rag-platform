import logging
from fastapi import APIRouter
from sqlalchemy import text
from app.database import SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health() -> dict:
    return {"status": "ok", "service": "rag-platform"}


@router.get("/db")
async def health_db() -> dict:
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return {"status": "error", "database": str(e)}


@router.get("/qdrant")
async def health_qdrant() -> dict:
    try:
        from qdrant_client import AsyncQdrantClient
        from app.config import get_settings

        settings = get_settings()
        client = AsyncQdrantClient(url=settings.qdrant_url)
        info = await client.get_collections()
        await client.close()
        return {"status": "ok", "qdrant": "connected", "collections": len(info.collections)}
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        return {"status": "error", "qdrant": str(e)}
