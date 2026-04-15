from fastapi import FastAPI
from fastapi import HTTPException
from typing import List
import logging


app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    import joblib
    model = joblib.load("model.pkl")
    
class PredictionResponse(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionFeatures):
    prediction = model.predict([[data.feature1, data.feature2]])
    return {"prediction": int(prediction[0])}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionFeatures):
    try:
        prediction = model.predict([[data.feature1, data.feature2]])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def preprocess(data: PredictionFeatures):
    return [[data.feature1, data.feature2]]  # ici trivial, mais extensible

X = preprocess(data)
prediction = model.predict(X)


@app.post("/predict_batch")
def predict_batch(data: List[PredictionFeatures]):
    X = [[d.feature1, d.feature2] for d in data]
    preds = model.predict(X)
    return {"predictions": [int(p) for p in preds]}


logging.basicConfig(level=logging.INFO)

@app.post("/predict")
def predict(data: PredictionFeatures):
    logging.info(f"Requête reçue: {data}")
    prediction = model.predict([[data.feature1, data.feature2]])
    return {"prediction": int(prediction[0])}
