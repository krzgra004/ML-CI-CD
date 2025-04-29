import os
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

X_train = np.array([[0.5], [14.0], [15.0], [28.0], [11.0], [8.0], [3.0], [-4.0], [6.0], [13.0], [21.0]])
y_train = np.array([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

model = LinearRegression()
model.fit(X_train, y_train)

class InputData(BaseModel):
    value: float

@app.get("/")
def read_root():
    project_id = os.getenv("PROJECT_ID", "Unknown")
    return {"Helllo": "Worlld", "project_id": project_id}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        prediction = model.predict(np.array([[input_data.value]]))
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
