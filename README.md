# RAG Platform

> **The problem:** "Our internal AI chatbot hallucinates and employees don't trust it. We can't tell when or why it's wrong."

A production-grade, multi-tenant Retrieval-Augmented Generation platform that makes your AI trustworthy. Every answer comes with source citations, a confidence score, and full LLM observability via Langfuse — so you always know *what* the model said, *why* it said it, and *how confident* it was.

---

## Architecture

```
INGESTION PIPELINE
──────────────────
  File Upload (PDF / TXT)
       │
       ▼
  [Chunker] ──► word-based sliding window, sentence-boundary aware
       │
       ▼
  [Embedder] ──► OpenAI text-embedding-3-small (1536-dim)
                 └─ in-process cache (sha256 keyed)
       │
       ▼
  [Qdrant] ──► collection: org_{org_id}   ← per-org isolation
  [Postgres] ──► documents table, status tracking

QUERY PIPELINE
──────────────
  User Query
       │
       ▼
  [Embedder] ──► embed query
       │
       ▼
  [Qdrant] ──► cosine similarity search in org_{org_id}
       │                 (NEVER crosses org boundaries)
       ▼
  [Confidence] ──► mean(top-3 scores) → 0.0–1.0
       │            if < 0.75 → low_confidence: true
       ▼
  [Generator] ──► GPT-4o-mini + numbered context chunks
       │          System: "Answer ONLY from context. Cite [1] [2]..."
       ▼
  [Citation Parser] ──► parse [N] refs from answer
       │                 return ONLY cited chunks (trust signal)
       ▼
  [Langfuse] ──► trace: retrieval span + generation span
       │
       ▼
  [QueryLog] ──► saved to Postgres (latency, tokens, sources)
       │
       ▼
  Response: { answer, sources, confidence, low_confidence, trace_id }
```

---

## Multi-Tenancy

Each organization gets its own Qdrant collection named `org_{org_id}`. This is enforced at the service layer:

- **Qdrant**: `QdrantService._collection_name(org_id)` returns `org_{org_id}` — every search, upsert, and delete targets only that collection.
- **Postgres**: All queries filter by `org_id` (FK enforced at DB level too).
- **API**: Routes are `/{org_id}/documents` and `/{org_id}/query` — the org is explicit in every request.

A query to Org A **cannot** return documents from Org B. The isolation is structural, not just policy.

---

## Quickstart

### With Docker Compose (recommended)

```bash
git clone https://github.com/Hemanthtr954/rag-platform.git
cd rag-platform

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY at minimum

docker compose up --build
```

The app is now at `http://localhost:8000`. Qdrant UI at `http://localhost:6333/dashboard`.

### Local development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in OPENAI_API_KEY, ensure Qdrant + Postgres are running

uvicorn app.main:app --reload
```

---

## API Reference

### 1. Create an Organization

```bash
curl -X POST http://localhost:8000/orgs \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp", "slug": "acme"}'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Acme Corp",
  "slug": "acme",
  "created_at": "2024-01-15T10:00:00Z"
}
```

### 2. Upload a Document

```bash
# PDF
curl -X POST http://localhost:8000/orgs/550e8400-e29b-41d4-a716-446655440000/documents \
  -F "file=@employee-handbook.pdf"

# TXT
curl -X POST http://localhost:8000/orgs/550e8400-e29b-41d4-a716-446655440000/documents \
  -F "file=@policy.txt"
```

Response (HTTP 202 — processing happens in background):
```json
{
  "id": "doc-uuid",
  "org_id": "550e8400-...",
  "filename": "employee-handbook.pdf",
  "content_hash": "abc123...",
  "chunk_count": 0,
  "status": "processing",
  "created_at": "2024-01-15T10:01:00Z"
}
```

Poll `GET /orgs/{org_id}/documents` until `status: "ready"`.

### 3. Query

```bash
curl -X POST http://localhost:8000/orgs/550e8400-e29b-41d4-a716-446655440000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the termination clause?", "top_k": 5}'
```

Response:
```json
{
  "answer": "According to the employee handbook [1], employees may be terminated with 30 days written notice [2]. Severance is calculated at one week per year of service [1].",
  "sources": [
    {
      "doc_id": "doc-uuid",
      "chunk_id": "doc-uuid_chunk_4",
      "filename": "employee-handbook.pdf",
      "excerpt": "Termination requires 30 days written notice...",
      "score": 0.9231,
      "citation_number": 1
    },
    {
      "doc_id": "doc-uuid",
      "chunk_id": "doc-uuid_chunk_7",
      "filename": "employee-handbook.pdf",
      "excerpt": "Written notice must be submitted to HR...",
      "score": 0.8876,
      "citation_number": 2
    }
  ],
  "confidence": 0.8931,
  "low_confidence": false,
  "trace_id": "trace-uuid",
  "latency_ms": 842
}
```

### 4. List Documents

```bash
curl http://localhost:8000/orgs/550e8400-e29b-41d4-a716-446655440000/documents
```

### 5. Delete a Document

```bash
curl -X DELETE http://localhost:8000/orgs/{org_id}/documents/{doc_id}
```

### 6. Health Checks

```bash
curl http://localhost:8000/health          # service status
curl http://localhost:8000/health/db       # postgres connectivity
curl http://localhost:8000/health/qdrant   # qdrant connectivity
```

---

## How Confidence Scoring Works

Confidence is a signal for how well the retrieved documents match the query.

**Formula:**
```
confidence = mean(score_1, score_2, score_3)   # top-3 Qdrant cosine similarity scores
```

- Scores range from 0.0 (no similarity) to 1.0 (identical)
- If `confidence < 0.75` → `low_confidence: true` in the response
- **Frontend should display a warning** when `low_confidence: true` — the model may be hallucinating or the topic isn't covered in the uploaded documents

**Why mean of top-3 (not top-1)?**
A single fluke match can score high. Averaging the top-3 gives a more robust signal of whether the knowledge base genuinely covers the topic.

---

## How Citation Tracking Works

After the LLM generates an answer, we parse `[1]`, `[2]`, etc. from the text.

**The key trust signal:** We only return sources that were *actually cited in the answer* — not all retrieved chunks.

This means:
- If 5 chunks were retrieved but the model only used chunks 1 and 3 → only those 2 appear in `sources`
- If the model says "I cannot find the answer..." → `sources: []`
- Users see exactly which documents contributed to the answer

This prevents the common anti-pattern of returning all retrieved chunks as "sources" even when the model ignored them.

---

## Document Deduplication

Before processing, the platform SHA-256 hashes the file content. If a document with that hash already exists in the org, you receive:

```json
HTTP 409 Conflict
{
  "detail": "A document with identical content already exists in this organization (hash: abc123def456...)"
}
```

This prevents accidental re-uploads from inflating the knowledge base.

---

## Langfuse Observability (Optional)

Langfuse provides a dashboard to inspect every LLM call: prompt, response, tokens, latency, and confidence.

**The app works without Langfuse configured** — all Langfuse calls fail gracefully with a warning log.

To enable:

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) (free tier available)
2. Create a project → copy Public Key and Secret Key
3. Add to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

Each query creates a Langfuse trace with:
- **Retrieval span**: query, result count, scores, latency
- **Generation span**: full prompt, response, token counts, confidence

The `trace_id` in the API response links directly to the Langfuse dashboard entry.

---

## Running Tests

No real API keys or running services required — all external dependencies are mocked.

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `DATABASE_URL` | Yes | SQLite | Postgres or SQLite URL |
| `QDRANT_URL` | Yes | `http://localhost:6333` | Qdrant server URL |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse secret key |
| `LANGFUSE_HOST` | No | cloud.langfuse.com | Langfuse host |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | No | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | No | `512` | Words per chunk |
| `CHUNK_OVERLAP` | No | `64` | Overlap words between chunks |
| `TOP_K` | No | `5` | Default retrieval results |
| `CONFIDENCE_THRESHOLD` | No | `0.75` | Low confidence threshold |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Vector DB | Qdrant (per-org collections) |
| Relational DB | PostgreSQL / SQLite |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4o-mini |
| Observability | Langfuse |
| ORM | SQLAlchemy (async) |
| Validation | Pydantic v2 |
| PDF parsing | pypdf |

---

## Why This Solves the Hallucination Trust Problem

1. **Citation tracking** — users see exactly which document each claim came from. No more "the AI said so."
2. **Confidence scores** — the frontend can warn users when the knowledge base doesn't have a good answer, instead of confidently hallucinating.
3. **Langfuse traces** — every LLM call is logged with full prompt and response. You can audit *why* a specific answer was generated.
4. **Per-org isolation** — cross-contamination between tenants is structurally impossible.
5. **Document dedup** — the knowledge base stays clean.

---

*Built with FastAPI, Qdrant, OpenAI, and Langfuse.*  
*The architecture that makes AI trustworthy enough to deploy internally.*
