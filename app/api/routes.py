from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.core.database import get_db

from app.schemas.asteroid_dto import AsteroidInput, AsteroidOutput
from app.schemas.image_dto import CelestialBodyPredictionOutput
from app.schemas.rl_dto import RivalTrainingRequest, RivalTrainingResponse, RivalPlayResponse

from app.services.asteroid_service import AsteroidService
from app.services.image_service import ImageService
from app.services.mars_service import MarsService

from app.worker.training_tasks import train_asteroid_model_task
from app.worker.training_tasks import train_mars_agents_task

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

rival_service = MarsService()
@router.post("/train/rivals", response_model=RivalTrainingResponse)
def train_rivals_endpoint(request: RivalTrainingRequest):
    """
    Launch the training Rover vs Saboteur with custom grid.
    """
    task = train_mars_agents_task.delay(request.grid_size, request.episodes)
    return RivalTrainingResponse(
        message=f"Rival training started on {request.grid_size}x{request.grid_size} grid.",
        task_id=task.id
    )

@router.get("/play/rivals", response_model=RivalPlayResponse)
def play_rivals_endpoint(grid_size: int = 8):
    """
    Watch a match between the Rover and the Saboteur.
    """
    result = rival_service.play_match(grid_size)
    if "error" in result:
        return RivalPlayResponse(
             grid_size=grid_size, start=(0,0), target=(0,0), initial_rocks=[],
             history=[], winner="Error: Model not found.", total_steps=0
        )
    return RivalPlayResponse(**result)