from fastapi import FastAPI

app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    import joblib
    model = joblib.load("model.pkl")
