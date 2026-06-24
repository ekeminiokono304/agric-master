from fastapi import APIRouter, HTTPException
from app.schemas.price_schema import PriceForecastResponse
from app.models.manager import model_manager
import numpy as np

router = APIRouter()

@router.get("/{commodity}", response_model=PriceForecastResponse)
async def forecast_prices(commodity: str):
    if commodity.lower() not in ['maize', 'rice']:
        raise HTTPException(status_code=400, detail="Commodity unsupported. Choose 'maize' or 'rice'.")
        
    try:
        pack = model_manager.get_model('price')
    except Exception:
        raise HTTPException(status_code=503, detail="Forecasting subsystem offline.")

    try:
        # Autoregressive Loop Construction
        current_sequence = list(pack['latest_sequence'])
        predictions = []
        
        for _ in range(30):
            input_vector = np.array(current_sequence[-7:]).reshape(1, -1)
            next_pred = pack['model'].predict(input_vector)[0]
            predictions.append(float(next_pred))
            current_sequence.append(next_pred)
            
        return PriceForecastResponse(
            commodity=commodity.capitalize(),
            current_baseline=float(current_sequence[6]),
            forecast_30_days_ahead=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))