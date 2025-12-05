from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class AsteroidInput(BaseModel):
    """
    Data Transfer Object for Asteroid input features.
    Matches the dataset columns.
    """
    absolute_magnitude: float = Field(..., description="Absolute Magnitude (H).")
    diameter_min_km: float = Field(..., gt=0, description="Minimum estimated diameter in km.")
    diameter_max_km: float = Field(..., gt=0, description="Maximum estimated diameter in km.")
    semi_major_axis: float = Field(..., description="Semi-major axis (au).")
    inclination: float = Field(..., description="Inclination (deg).")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "absolute_magnitude": 22.5,
                "diameter_min_km": 0.12,
                "diameter_max_km": 0.28,
                "semi_major_axis": 1.45,
                "inclination": 5.8
            }
        }
    }

class AsteroidOutput(AsteroidInput):
    """
    Output schema including prediction results and metadata.
    """
    id: int
    predicted_hazardous: bool
    confidence_score: Optional[float] = None
    impact_probability: float = Field(..., description="Probability of Earth impact (0.0 to 1.0).")
    model_version: str
    created_at: datetime

    model_config = {
        "protected_namespaces": (),
        "from_attributes": True
    }