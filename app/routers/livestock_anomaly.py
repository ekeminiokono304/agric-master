from fastapi import APIRouter, HTTPException
from app.schemas.livestock_schema import LivestockTelemetryInput, LivestockAnomalyResponse
from app.models.manager import model_manager
import numpy as np

router = APIRouter()

@router.post("/", response_model=LivestockAnomalyResponse)
async def check_telemetry(payload: LivestockTelemetryInput):
    try:
        model = model_manager.get_model('livestock')
    except Exception:
        raise HTTPException(status_code=503, detail="Telemetry analysis engine unavailable.")
        
    try:
        features = np.array([[payload.heart_rate, payload.temperature, payload.activity_level]])
        prediction = model.predict(features)[0] # Returns 1 for normal, -1 for anomaly
        score = model.score_samples(features)[0]
        
        is_anomaly = True if prediction == -1 else False
        rec = "Status Normal. Continuous monitoring maintained."
        if is_anomaly:
            rec = "Critical Alert: Deviant biometric signature noted. Veterinary intervention advised."
            
        return LivestockAnomalyResponse(
            animal_id=payload.animal_id,
            is_anomaly=is_anomaly,
            anomaly_score=float(score),
            recommendation=rec
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))