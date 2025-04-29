import os
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
from model import train_and_predict, get_accuracy

#test

app = FastAPI()

class InputData(BaseModel):
    value: float

@app.get("/")
def read_root():
    project_id = os.getenv("PROJECT_ID", "Unknown")
    return {"Helllo": "Worlld", "project_id": project_id}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        prediction, _ = train_and_predict
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/info")
def model_info():
    return {
        "typ modelu": "LinearRegression",
        "cechy": 1,
        "ilość próbek": len(X_train)
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
