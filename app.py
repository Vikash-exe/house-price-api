from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

model = joblib.load("house_price_rf_pipeline.pkl")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"predicted_price": float(pred[0])}
