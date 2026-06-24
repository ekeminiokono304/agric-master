from fastapi import APIRouter, HTTPException
from app.schemas.drought_schema import DroughtRiskInput, DroughtRiskResponse
from app.models.manager import model_manager
import numpy as np

router = APIRouter()
RISK_MAP = {0: "Low Risk Profile", 1: "Moderate Risk Warning", 2: "Severe Critical Status"}

@router.post("/", response_model=DroughtRiskResponse)
async def assess_drought(payload: DroughtRiskInput):
    try:
        model = model_manager.get_model('drought')
    except Exception:
        raise HTTPException(status_code=503, detail="Drought monitoring service offline.")
        
    try:
        features = np.array([[payload.rainfall_mm, payload.temperature_c, payload.soil_moisture]])
        pred_class = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        
        prob_dict = {RISK_MAP[i]: float(proba[i]) for i in range(len(RISK_MAP))}
        
        return DroughtRiskResponse(
            risk_level=RISK_MAP[pred_class],
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))