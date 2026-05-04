from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import Ridge
import json

app = FastAPI()

# Input schema
class InputData(BaseModel):
    rock_hardness: float
    drill_bit_age_hours: float
    mud_pressure_psi: float
    depth_m: float

# Train model once
df = pd.read_csv("data/training_data.csv")
X = df.drop("drilling_rate_m_per_hr", axis=1)
y = df["drilling_rate_m_per_hr"]

model = Ridge(alpha=1.0)
model.fit(X, y)

# Health endpoint
@app.get("/heartbeat")
def health():
    return {
        "status": "running",
        "model": "Ridge",
        "version": "1.0"
    }

# Prediction endpoint
@app.post("/infer")
def predict(data: InputData):
    try:
        features = [[
            data.rock_hardness,
            data.drill_bit_age_hours,
            data.mud_pressure_psi,
            data.depth_m
        ]]
        
        pred = model.predict(features)[0]

        result = {
            "input": data.dict(),
            "prediction": float(pred)
        }

        # Save result
        with open("results/step4_s4.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))