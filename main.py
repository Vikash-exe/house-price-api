from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("house_price_xgb.pkl")

@app.get("/")
def health():
    return {"status": "ok"}

class HouseInput(BaseModel):
    Location: str
    Size: float
    Bedrooms: int
    Bathrooms: int
    Condition: str
    Type: str
    inflation_time: int
    Age: int

@app.post("/predict")
def predict(data: HouseInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}
