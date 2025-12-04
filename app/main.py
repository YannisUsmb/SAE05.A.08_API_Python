from typing import Union

from fastapi import FastAPI
from app.core.database import engine, Base
from app.models_db import AsteroidPrediction
from app.api.routes import router

# Create tables.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI API")

# Register the router (Controller).
app.include_router(router, prefix="/api/v1", tags=["Predictions"])

@app.get("/")
def read_root():
    return {"status": "online", "message": "API is running."}

@app.get("/health")
def health_check():
    # Later, add a connexion test to redis.
    return {"db": "ok"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}