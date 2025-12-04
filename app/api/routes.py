from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.asteroid_dto import AsteroidInput, AsteroidOutput
from app.services.asteroid_service import AsteroidService

router = APIRouter()

@router.post("/predict/asteroid", response_model=AsteroidOutput, status_code=201)
def predict_asteroid(payload: AsteroidInput, db: Session = Depends(get_db)):
    """
    Predict if an asteroid is potentially hazardous based on its features.
    """
    try:
        # Instantiate the Service with the DB session.
        service = AsteroidService(db)
        
        # Run business logic.
        result = service.predict_impact(payload)
        
        return result
    
    except Exception as e:
        # Log the error.
        raise HTTPException(status_code=500, detail=str(e))