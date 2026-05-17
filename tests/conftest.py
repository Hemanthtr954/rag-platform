import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Use SQLite for tests — no real services needed
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
