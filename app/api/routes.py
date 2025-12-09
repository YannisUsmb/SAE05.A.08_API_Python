from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.asteroid_dto import AsteroidInput, AsteroidOutput
from app.schemas.image_dto import CelestialBodyPredictionOutput
from app.services.asteroid_service import AsteroidService
from app.services.image_service import ImageService
from app.worker.training_tasks import train_asteroid_model_task, train_celestialbody_model_task

router = APIRouter()

@router.post("/predict-asteroid", response_model=AsteroidOutput, status_code=201)
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
    
@router.post("/train/asteroid-model", status_code=202)
def trigger_training():
    """
    Triggers the training process in the background Worker.
    """
    task = train_asteroid_model_task.delay()
    return {
        "message": "Training started in background.",
        "task_id": task.id
    }

@router.post("/train/image-model")
def trigger_image_training():
    """
    Triggers the download/training of the ResNet image model.
    """
    task = train_celestialbody_model_task.delay()
    return {
        "message": "Image model training started in background.",
        "task_id": task.id
    }

@router.post("/predict-image", response_model=CelestialBodyPredictionOutput)
def predict_image(
    file: UploadFile = File(...),
    model_version_id: str = Form(None)
):
    """
    Upload an image (JPG/PNG) to classify it.
    """
    # Lire les octets du fichier
    file_bytes = file.file.read()
    
    service = ImageService()
    result = service.predict(
        file_bytes=file_bytes, 
        filename=file.filename, 
        content_type=file.content_type,
        version_id=model_version_id
    )
    
    return result