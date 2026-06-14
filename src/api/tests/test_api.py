from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check_returns_200():
    """Check that k8s probe answers correctly."""
    response = client.get("/health")

    # HTTP status assert
    assert response.status_code == 200
    # JSON assert
    assert response.json() == {"status": "ok", "service": "fleet_brain_api"}

def test_dispatch_command_accepts_valid_payload():
    """Check that API accepts a well-formed command."""
    payload = {
        "user_id": "test_runner",
        "instruction": "Esplora il perimetro"
    }

    response = client.post("/api/v1/command", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert "task_id" in data

def test_dispatch_command_rejects_invalid_payload():
    """Check that Pydantic blocks blocchi wrong data (missing 'instruction')."""
    bad_payload = {
        "user_id": "test_runner"
    }

    response = client.post("/api/v1/command", json=bad_payload)

    # We expect a 422 error (Unprocessable Entity) from FastAPI
    assert response.status_code == 422
