from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import main  # importer le module complet

client = TestClient(main.app)

API_KEY = "secret"

# ------------------------
# Mock du modèle
# ------------------------
def setup_module():
    main.app.router.on_startup.clear()

    mock_model = MagicMock()

    def _predict(X):
        if not X:
            return []
        return [1] * len(X)

    mock_model.predict.side_effect = _predict
    main.model = mock_model


# ------------------------
# Test racine
# ------------------------
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur mon API ML avec FastAPI !"}


# ------------------------
# Test prédiction OK
# ------------------------
def test_predict_success():
    response = client.post(
        "/predict",
        json={"feature1": 1.5, "feature2": 2.5},
        headers={"x-api-key": API_KEY},
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)


# ------------------------
# Test sans API key
# ------------------------
def test_predict_no_api_key():
    response = client.post(
        "/predict",
        json={"feature1": 1.5, "feature2": 2.5},
    )

    assert response.status_code == 422 or response.status_code == 403


# ------------------------
# Test API key invalide
# ------------------------
def test_predict_wrong_api_key():
    response = client.post(
        "/predict",
        json={"feature1": 1.5, "feature2": 2.5},
        headers={"x-api-key": "wrong_key"},
    )

    assert response.status_code == 403


# ------------------------
# Test données invalides
# ------------------------
def test_predict_invalid_data():
    response = client.post(
        "/predict",
        json={"feature1": "invalid", "feature2": 2.5},
        headers={"x-api-key": API_KEY},
    )

    assert response.status_code == 422


# ------------------------
# Test batch OK
# ------------------------
def test_predict_batch_success():
    response = client.post(
        "/predict_batch",
        json=[
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0},
        ],
        headers={"x-api-key": API_KEY},
    )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert all(isinstance(p, int) for p in data["predictions"])


# ------------------------
# Test batch vide
# ------------------------
def test_predict_batch_empty():
    response = client.post(
        "/predict_batch",
        json=[],
        headers={"x-api-key": API_KEY},
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": []}
