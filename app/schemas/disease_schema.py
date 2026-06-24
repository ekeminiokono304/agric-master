from pydantic import BaseModel
from typing import List

class DiseaseClassProba(BaseModel):
    disease_class: str
    probability: float

class DiseaseResponse(BaseModel):
    top_prediction: str
    severity_assessment: str
    probabilities: List[DiseaseClassProba]