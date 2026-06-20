import pytest
from unittest.mock import AsyncMock, MagicMock
from main import state

# --- STUB CLASSES FOR ASYNCPG ---
# These classes act as real objects to satisfy Python's async context manager protocol.
# This prevents the "coroutine object does not support..." error.

class MockConnection:
    # Add other DB methods you use here (e.g., fetch, fetchrow)
    async def execute(self, *args, **kwargs):
        return None

class MockPool:
    def acquire(self):
        # Returns self as the context manager
        return self

    async def __aenter__(self):
        # Called when entering the 'async with' block
        return MockConnection()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Called when exiting the 'async with' block
        pass

# --- SETUP FIXTURE ---

@pytest.fixture(scope="session", autouse=True)
def setup_mocks():
    """
    Injects mocks and stubs once per session to isolate the API from 
    external infrastructure (DB, Qdrant, ML models).
    """
    # 1. Use the stub class for the database pool
    state.pg_pool = MockPool()

    # 2. Mock Qdrant client
    state.qdrant_client = AsyncMock()

    # 3. Mock CLIP Model and Processor
    mock_model = MagicMock()
    # Mocking the embedding chain for the CLIP model
    mock_model.text_model.return_value.pooler_output.detach.return_value.cpu.return_value.reshape.return_value.tolist.return_value = [0.1] * 512

    mock_processor = MagicMock(return_value={
        "input_ids": MagicMock(),
        "attention_mask": MagicMock()
    })

    state.ml_models = {
        "clip_model": mock_model,
        "clip_processor": mock_processor
    }

    # 4. Force CPU mode to avoid CUDA initialization errors in CI/Testing
    state.device = "cpu"