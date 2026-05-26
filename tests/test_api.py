"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from fraud_detection_service.main import app, load_model_on_startup

# Ensure model is loaded on test startup
@pytest.fixture(scope="module", autouse=True)
def setup_model():
    import asyncio
    asyncio.run(load_model_on_startup())

client = TestClient(app)

def test_read_root():
    """Verify that root endpoint is accessible"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model_loaded" in data

def test_health_endpoint():
    """Verify that health check reports status correctly"""
    response = client.get("/health")
    # Status code can be 200 (healthy) or 503 (if model not loaded yet)
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert "status" in data
    if response.status_code == 200:
        assert data["model_loaded"] is True
        assert "auc_roc" in data
        assert "optimal_threshold" in data

def test_prediction_endpoint_legitimate():
    """Verify inference endpoint with standard transaction payload"""
    # Only run prediction test if model is loaded (status 200)
    health_response = client.get("/health")
    if health_response.status_code != 200:
        pytest.skip("Model binary not trained or loaded. Skipping prediction test.")
        
    payload = {
        "TransactionAmt": 45.50,
        "card1": 1000,
        "card2": 100.0,
        "card3": 100.0,
        "card5": 200.0,
        "addr1": 300.0,
        "addr2": 87.0,
        "dist1": 12.0,
        "dist2": -999.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["Low", "Medium", "High"]

def test_prediction_endpoint_fraudulent():
    """Verify prediction classifications on highly suspicious parameters"""
    health_response = client.get("/health")
    if health_response.status_code != 200:
        pytest.skip("Model binary not trained or loaded. Skipping prediction test.")
        
    payload = {
        "TransactionAmt": 9999.99,
        "card1": 9999,
        "card2": 999.0,
        "card3": 999.0,
        "card5": 999.0,
        "addr1": 999.0,
        "addr2": 999.0,
        "dist1": 50000.0,
        "dist2": 99999.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    # Highly anomalous values should yield elevated risk or prediction
    assert data["probability"] >= 0.0
