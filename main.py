from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List
import logging
import joblib

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
    feature1: float
    feature2: float

class PredictionResponse(BaseModel):
    prediction: int

class BatchPredictionResponse(BaseModel):
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
    try:
        logging.info(f"Requête reçue: {data}")

        X = preprocess(data)
        prediction = model.predict(X)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logging.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_key)])
def predict_batch(data: List[PredictionFeatures]):
    try:
        X = [[d.feature1, d.feature2] for d in data]
        preds = model.predict(X)

        return {"predictions": [int(p) for p in preds]}

    except Exception as e:
        logging.error(f"Erreur batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))
