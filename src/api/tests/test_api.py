import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app, state

# Initialize the TestClient
client = TestClient(app)

def test_health_check():
    """Verify that the FastAPI application mounts correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# Mock the embedding generation to avoid loading the heavy CLIP model during tests
@patch("main.get_embedding")
# Mock the LLM network call. Note: new_callable=AsyncMock is required because get_agent_plan is async
@patch("main.get_agent_plan", new_callable=AsyncMock)
def test_dispatch_command_success(mock_get_agent_plan, mock_get_embedding):
    """
    Test the main /command endpoint.
    Verifies that a text command is processed, parsed by the mock LLM,
    and correctly pushed to the Redis task queue.
    """
    # 1. ARRANGE: Setup mocks and test data

    # Return a dummy 512-dimensional vector to bypass the ML inference
    mock_get_embedding.return_value = [0.1] * 512

    # Force a deterministic JSON output from the agent
    mock_get_agent_plan.return_value = [
        {"action": "NAVIGATE", "target": "charger_station"}
    ]

    # Inject an AsyncMock for Redis into the global app state
    # This prevents real network calls to the Redis broker
    state.redis_client = AsyncMock()

    # Payload matching the CommandRequest Pydantic model
    payload = {
        "user_id": "operator_1",
        "instruction": "Find where the charger is and go there"
    }

    # 2. ACT: Perform the POST request
    response = client.post("/api/v1/command", json=payload)

    # 3. ASSERT: Validate API behavior and external integrations
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "accepted"
    assert len(data["plan"]) == 1
    assert data["plan"][0]["action"] == "NAVIGATE"

    # Verify the LLM was called with the exact instruction string
    mock_get_agent_plan.assert_called_once_with("Find where the charger is and go there")

    # Verify the Redis rpush command was triggered exactly once
    state.redis_client.rpush.assert_called_once()

    # Extract the arguments passed to Redis to ensure the payload is correct
    queue_name, pushed_json = state.redis_client.rpush.call_args[0]

    assert queue_name == "robot_tasks_queue"
    assert "NAVIGATE" in pushed_json
    assert "operator_1" in pushed_json