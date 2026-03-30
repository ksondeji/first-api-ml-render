from fastapi import FastAPI
from pydantic import BaseModel # Importez BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

# Définir le "contrat" de données
class PredictionFeatures(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(data: PredictionFeatures): # Utilisez le modèle comme type
    prediction = model.predict([[data.feature1, data.feature2]])
    return {"prediction": int(prediction[0])}
    
# 1. Initialiser l'application FastAPI
app = FastAPI(
    title="Mon API de Machine Learning",
    description="Une API simple pour tester le déploiement sur Render."
)

# 2. Charger le modèle (assurez-vous qu'il est dans votre repo)
# Pour cet exemple, nous créons un modèle factice s'il n'existe pas.
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit([[0], [1]], [0, 1]) # Entraînement factice

# 3. Définir le premier endpoint (la racine)
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API ML !"}

# 4. Définir l'endpoint de prédiction
@app.post("/predict")
def predict(feature1: float, feature2: float):
    # Logique de prédiction simple
    prediction = model.predict([[feature1, feature2]])
    return {"prediction": int(prediction[0])}

