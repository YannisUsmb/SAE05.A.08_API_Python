from sqlalchemy.orm import Session
from app.models_db import AsteroidPrediction
from app.schemas.asteroid_dto import AsteroidInput

class AsteroidRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_prediction(
        self, 
        input_data: AsteroidInput, 
        is_hazardous: bool, 
        confidence: float, 
        impact_prob: float,
        model_version: str
    ) -> AsteroidPrediction:
        """
        Saves a new inference result into the database.
        """
        db_prediction = AsteroidPrediction(
            # Mapping input features.
            absolute_magnitude=input_data.absolute_magnitude,
            diameter_min_km=input_data.diameter_min_km,
            diameter_max_km=input_data.diameter_max_km,
            semi_major_axis=input_data.semi_major_axis,
            inclination=input_data.inclination,
            
            # Mapping prediction results.
            predicted_hazardous=is_hazardous,
            confidence_score=confidence,
            impact_probability=impact_prob,
            model_version=model_version
        )
        
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction)
        return db_prediction

    def get_recent_predictions(self, limit: int = 10):
        """
        Retrieves the last N predictions.
        """
        return self.db.query(AsteroidPrediction)\
            .order_by(AsteroidPrediction.created_at.desc())\
            .limit(limit)\
            .all()