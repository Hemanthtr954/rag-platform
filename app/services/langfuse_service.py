import logging
import uuid
from app.config import get_settings

logger = logging.getLogger(__name__)


class LangfuseService:
    """
    Langfuse observability wrapper.
    Fails gracefully when Langfuse is not configured — the app works without it.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = None
        self._enabled = False

        if self._settings.langfuse_enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self._settings.langfuse_public_key,
                    secret_key=self._settings.langfuse_secret_key,
                    host=self._settings.langfuse_host,
                )
                self._enabled = True
                logger.info("Langfuse observability enabled")
            except Exception as e:
                logger.warning("Failed to initialize Langfuse: %s — continuing without it", e)
        else:
            logger.info("Langfuse not configured — skipping observability (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable)")

    def create_trace(self, name: str, org_id: str, query: str) -> str:
        """Create a new Langfuse trace. Returns trace_id (generated locally if Langfuse disabled)."""
        trace_id = str(uuid.uuid4())
        if not self._enabled or self._client is None:
            return trace_id

        try:
            trace = self._client.trace(
                id=trace_id,
                name=name,
                metadata={"org_id": org_id},
                input=query,
            )
            return trace.id
        except Exception as e:
            logger.warning("Langfuse create_trace failed: %s", e)
            return trace_id

    def log_retrieval(
        self,
        trace_id: str,
        query: str,
        results: list,
        latency_ms: int,
    ) -> None:
        """Log retrieval span to Langfuse."""
        if not self._enabled or self._client is None:
            return

        try:
            self._client.span(
                trace_id=trace_id,
                name="retrieval",
                input={"query": query},
                output={"result_count": len(results), "scores": [r.score for r in results]},
                metadata={"latency_ms": latency_ms},
            )
        except Exception as e:
            logger.warning("Langfuse log_retrieval failed: %s", e)

    def log_generation(
        self,
        trace_id: str,
        prompt: str,
        response: str,
        tokens: dict,
        latency_ms: int,
        confidence: float,
    ) -> None:
        """Log LLM generation span to Langfuse."""
        if not self._enabled or self._client is None:
            return

        try:
            self._client.generation(
                trace_id=trace_id,
                name="generation",
                input=prompt,
                output=response,
                usage={
                    "prompt_tokens": tokens.get("prompt_tokens", 0),
                    "completion_tokens": tokens.get("completion_tokens", 0),
                    "total_tokens": tokens.get("prompt_tokens", 0) + tokens.get("completion_tokens", 0),
                },
                metadata={"latency_ms": latency_ms, "confidence": confidence},
            )
        except Exception as e:
            logger.warning("Langfuse log_generation failed: %s", e)

    def flush(self) -> None:
        """Flush all pending events to Langfuse."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)
