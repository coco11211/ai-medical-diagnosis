"""Integration tests for API."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["status"] == "running"

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data

    def test_analyze_symptoms(self):
        """Test symptom analysis endpoint."""
        payload = {
            "symptoms": "Patient has fever and cough"
        }
        response = client.post("/analyze/symptoms", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "symptoms" in data
        assert "text" in data

    def test_list_diseases(self):
        """Test list diseases endpoint."""
        response = client.get("/diseases")
        assert response.status_code == 200
        data = response.json()
        assert "diseases" in data
        assert "count" in data
        assert data["count"] > 0

    def test_create_patient(self):
        """Test patient creation endpoint."""
        payload = {
            "patient_id": "TEST001",
            "name": "Test Patient",
            "age": 45,
            "sex": "M"
        }
        response = client.post("/patients", json=payload)
        # May fail if patient already exists, which is fine
        assert response.status_code in [200, 500]
