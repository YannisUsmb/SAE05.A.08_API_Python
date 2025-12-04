from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.core.database import Base

class AsteroidPrediction(Base):
    __tablename__ = "asteroid_predictions"

    id = Column(Integer, primary_key=True, index=True)

    # Features.
    absolute_magnitude = Column(Float, nullable=False, comment="Feature: ast_absolutemagnitude")
    diameter_min_km = Column(Float, nullable=False, comment="Feature: ast_diametermin")
    diameter_max_km = Column(Float, nullable=False, comment="Feature: ast_diametermax")
    semi_major_axis = Column(Float, nullable=False, comment="Feature: ast_semimajoraxis")
    inclination = Column(Float, nullable=False, comment="Feature: ast_inclination")

    # Prediction.
    predicted_hazardous = Column(Boolean, comment="AI prediction (True/False).")
    confidence_score = Column(Float, comment="Confidence score (e.g: 0.95).")
    model_version = Column(String, default="v1", comment="AI model version used.")

    # Meta.
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Feedback.
    actual_hazardous = Column(Boolean, nullable=True, comment="Real result, validated by human.")