import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.routers import organizations, documents, query, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()
    logger.info("RAG Platform started")
    yield
    # Shutdown
    logger.info("RAG Platform shutting down")


app = FastAPI(
    title="RAG Platform",
    description=(
        "Multi-tenant Retrieval-Augmented Generation platform. "
        "Per-org data isolation, hybrid search, citation tracking, "
        "confidence scoring, and Langfuse observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(organizations.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/")
async def root() -> dict:
    return {
        "service": "RAG Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
