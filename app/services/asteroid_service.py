import random
from sqlalchemy.orm import Session
from app.repositories.asteroid_repository import AsteroidRepository
from app.schemas.asteroid_dto import AsteroidInput, AsteroidOutput

class AsteroidService:
    def __init__(self, db: Session):
        self.repository = AsteroidRepository(db)

    def predict_impact(self, input_data: AsteroidInput) -> AsteroidOutput:
        """
        Orchestrates the prediction flow:
        1. Prepares data.
        2. Loads model (or uses mock).
        3. Predicts.
        4. Saves result to DB.
        """
        
        # TODO: Load real LightGBM model here later.
        # For now, we simulate a prediction to test the API architecture.
        # Logic: If diameter > 0.15km and semi_major_axis < 1.0, it's risky.
        is_hazardous_simulated = (
            input_data.diameter_max_km > 0.15 and 
            input_data.semi_major_axis < 1.0
        )
        
        confidence_simulated = random.uniform(0.85, 0.99) if is_hazardous_simulated else random.uniform(0.01, 0.15)

        if is_hazardous_simulated:
            impact_prob_simulated = random.uniform(0.01, 0.05) 
        else:
            impact_prob_simulated = random.uniform(0.000001, 0.0001)
        
        model_version = "v0-mock"

        # Save to Database (Repository Layer).
        saved_prediction = self.repository.create_prediction(
            input_data=input_data,
            is_hazardous=is_hazardous_simulated,
            confidence=confidence_simulated,
            impact_prob=impact_prob_simulated,
            model_version=model_version
        )

        # Convert SQL Model to Pydantic DTO.
        return AsteroidOutput.model_validate(saved_prediction)