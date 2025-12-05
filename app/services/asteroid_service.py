import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.repositories.asteroid_repository import AsteroidRepository
from app.schemas.asteroid_dto import AsteroidInput, AsteroidOutput
from app.core.model_loader import ml_models

class AsteroidService:
    def __init__(self, db: Session):
        self.repository = AsteroidRepository(db)

    def predict_impact(self, input_data: AsteroidInput) -> AsteroidOutput:
        """
        Orchestrates the prediction flow:
        1. Prepares data.
        2. Loads model.
        3. Predicts.
        4. Saves result to DB.
        Strict mode: If model is missing, raises 503 error.
        """

        if ml_models.asteroid_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Asteroid Prediction Model is unavailable."
            )

        features = np.array([[
            input_data.absolute_magnitude,
            input_data.diameter_min_km,
            input_data.diameter_max_km,
            input_data.semi_major_axis,
            input_data.inclination
        ]])

        try:
            prob = ml_models.asteroid_model.predict(features)[0]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Asteroid Prediction Model Engine Error: {str(e)}"
            )
        
        is_hazardous = bool(prob > 0.5)
        confidence = float(prob) if is_hazardous else float(1 - prob)
        impact_prob = float(prob / 100)
        model_version = "v1-alpha"
            
        # Save to Database (Repository Layer).
        saved_prediction = self.repository.create_prediction(
            input_data=input_data,
            is_hazardous=is_hazardous,
            confidence=confidence,
            impact_prob=impact_prob,
            model_version=model_version
        )

        # Convert SQL Model to Pydantic DTO.
        return AsteroidOutput.model_validate(saved_prediction)