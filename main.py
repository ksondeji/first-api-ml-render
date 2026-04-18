import logging
import joblib
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI()

# ------------------------
# Configuration logging
# ------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------
# Sécurité (API Key)
# ------------------------
API_KEY = "secret"

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

# ------------------------
# Modèle ML
# ------------------------
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("model.pkl")
        logging.info("Modèle chargé avec succès")
    except Exception as e:
        logging.error(f"Erreur chargement modèle: {e}")
        raise e

# ------------------------
# Schémas Pydantic
# ------------------------
class PredictionFeatures(BaseModel):
    """Modèle pour les caractéristiques d'entrée de prédiction."""
    feature1: float
    feature2: float

class PredictionResponse(BaseModel):
    prediction: int

class BatchPredictionResponse(BaseModel):
    """Modèle pour la réponse de prédiction en lot."""
    predictions: List[int]

# ------------------------
# Préprocessing
# ------------------------
def preprocess(data: PredictionFeatures):
    return [[data.feature1, data.feature2]]

# ------------------------
# Routes
# ------------------------
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API ML avec FastAPI !"}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_key)])
def predict(data: PredictionFeatures):
    """
    Effectue une prédiction basée sur les caractéristiques fournies.

    Args:
        data: Objet PredictionFeatures avec les caractéristiques d'entrée.

    Returns:
        Dictionnaire avec la prédiction sous forme d'entier.

    Raises:
        HTTPException: En cas d'erreur interne.
    """
    try:
        logging.info("Requête reçue: %s", data)

        features = preprocess(data)
        prediction = model.predict(features)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logging.error("Erreur prédiction: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict_batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_key)])
def predict_batch(data: List[PredictionFeatures]):
    """
    Effectue des prédictions en lot sur une liste de caractéristiques.

    Args:
        data: Liste d'objets PredictionFeatures.

    Returns:
        Dictionnaire avec les prédictions sous forme d'entiers.

    Raises:
        HTTPException: En cas d'erreur interne.
    """
    try:
        features = [[d.feature1, d.feature2] for d in data]
        preds = model.predict(features)
        return {"predictions": [int(p) for p in preds]}
    except Exception as e:
        logging.error("Erreur batch: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
