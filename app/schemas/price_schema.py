from pydantic import BaseModel
from typing import List

class PriceForecastResponse(BaseModel):
    commodity: str
    current_baseline: float
    forecast_30_days_ahead: List[float]