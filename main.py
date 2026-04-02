from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Mon API de Machine Learning",
    description="Une API simple pour tester le déploiement sur Render.",
)

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])


class PredictionFeatures(BaseModel):
    feature1: float
    feature2: float


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API ML !"}


@app.post("/predict")
def predict(data: PredictionFeatures):
    prediction = model.predict([[data.feature1, data.feature2]])
    return {"prediction": int(prediction[0])}
