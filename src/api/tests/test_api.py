from fastapi.testclient import TestClient
from main import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

def test_dispatch_command_success():
    with TestClient(app) as client:
        payload = {"user_id": "test_runner", "instruction": "Esplora"}
        response = client.post("/api/v1/command", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

def test_test_embedding_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/test-embedding?text=test")
        assert response.status_code == 200
        assert response.json()["status"] == "success"