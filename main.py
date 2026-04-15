from fastapi import FastAPI

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
