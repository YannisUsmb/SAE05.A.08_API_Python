from fastapi import FastAPI
from app.core.database import engine, Base
from app.models_db import AsteroidPrediction
from app.api.routes import router
from app.core.model_loader import ml_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models.load_models()
    yield
    print("Shutting down API.")

# Create tables.
Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI API", lifespan=lifespan)

# Register the router (Controller).
app.include_router(router, prefix="/api/v1", tags=["Predictions"])

@app.get("/")
def read_root():
    return {"status": "online", "message": "API is running."}

@app.get("/health")
def health_check():
    # Later, add a connexion test to redis.
    return {"db": "ok"}