from fastapi import APIRouter, HTTPException
from app.schemas.yield_schema import YieldRequest, YieldResponse
from app.models.manager import model_manager
import pandas as pd
import numpy as np

router = APIRouter()

@router.post("/", response_model=YieldResponse)
async def predict_yield(payload: YieldRequest):
    try:
        pack = model_manager.get_model('yield')
    except Exception:
        raise HTTPException(status_code=503, detail="Yield model unavailable.")
        
    try:
        # Preprocessing
        input_data = {
            'soil_type': [payload.soil_type],
            'crop_type': [payload.crop_type],
            'region': [payload.region],
            'rainfall': [payload.rainfall],
            'fertilizer_use': [payload.fertilizer_use]
        }
        df = pd.DataFrame(input_data)
        
        for col, encoder in pack['encoders'].items():
            if df[col].iloc[0] in encoder.classes_:
                df[col] = encoder.transform(df[col])
            else:
                df[col] = 0 # Fallback for unseen values
                
        pred = pack['model'].predict(df)[0]
        
        # Simple analytic approximation for CI for demo purposes
        return YieldResponse(
            predicted_yield=float(pred),
            confidence_lower=float(pred - (pred * 0.08)),
            confidence_upper=float(pred + (pred * 0.08))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")