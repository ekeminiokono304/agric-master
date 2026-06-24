from pydantic import BaseModel, Field

class DroughtRiskInput(BaseModel):
    rainfall_mm: float = Field(..., example=34.2)
    temperature_c: float = Field(..., example=38.1)
    soil_moisture: float = Field(..., example=12.5)

class DroughtRiskResponse(BaseModel):
    risk_level: str
    probabilities: dict