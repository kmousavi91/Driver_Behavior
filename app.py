# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import datetime
import json
import os
import logging

# === Configuration ===
MODEL_PATH = "driver_behavior_model.pkl"
LABEL_MAP = {

  0: "Normal",
  1: "Aggressive",
  2: "Risky",
  3: "Drowsy",
  4: "Dangerous"

}

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("driver-api")

# === Load Model ===
try:
    model, feature_names = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model and features from: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# === Local Log File ===
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/predictions_{datetime.date.today()}.jsonl"

# === FastAPI App ===
app = FastAPI(title="Driver Behavior API", description="Predict driving behavior using 60 features")

class PredictionRequest(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict_behavior(request: PredictionRequest):
    try:
        if len(request.features) != len(feature_names):
            raise HTTPException(status_code=400, detail=f"Expected {len(feature_names)} features, got {len(request.features)}")

        prediction = model.predict([request.features])[0]
        predicted_label = LABEL_MAP.get(int(prediction), "Unknown")

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "features": request.features,
            "predicted_class": int(prediction),
            "predicted_label": predicted_label
        }

        with open(log_filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "predicted_class": int(prediction),
            "predicted_label": predicted_label
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Driver behavior prediction API is live."}

