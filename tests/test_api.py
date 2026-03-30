# tests/test_api.py
from fastapi.testclient import TestClient
from main import app # Importez votre app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur mon API ML !"}

def test_predict_endpoint():
    response = client.post("/predict", json={"feature1": 1.5, "feature2": 2.5})
    assert response.status_code == 200
    assert "prediction" in response.json()