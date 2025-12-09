from pydantic import BaseModel
from typing import Optional

class CelestialBodyPredictionOutput(BaseModel):
    filename: str
    content_type: str
    predicted_class: str
    confidence_score: float
    model_version: str
    
    model_config = {
        "protected_namespaces": (),
        "from_attributes": True
    }