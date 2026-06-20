from fastapi.testclient import TestClient
from main import app

def test_health_check_returns_200():
    """Check that the health check endpoint returns a 200 OK status."""
    # Use context manager to ensure lifespan (startup/shutdown events) is triggered
    with TestClient(app) as client:
        response = client.get("/health")

        # Verify status code
        assert response.status_code == 200
        # Verify JSON response body
        assert response.json() == {"status": "ok", "service": "fleet_brain_api"}

def test_dispatch_command_accepts_valid_payload():
    """Verify that the API correctly processes a valid command payload."""
    with TestClient(app) as client:
        payload = {
            "user_id": "test_runner",
            "instruction": "Esplora il perimetro"
        }

        response = client.post("/api/v1/command", json=payload)

        # Assert successful processing
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "task_id" in data

def test_dispatch_command_rejects_invalid_payload():
    """Verify that the API returns 422 Unprocessable Entity for missing required fields."""
    with TestClient(app) as client:
        # Instruction field is missing
        bad_payload = {
            "user_id": "test_runner"
        }

        response = client.post("/api/v1/command", json=bad_payload)

        # FastAPI automatically validates Pydantic models and returns 422
        assert response.status_code == 422