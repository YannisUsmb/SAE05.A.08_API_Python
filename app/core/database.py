from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Get URL of DB from docker-compose.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:postgres@localhost:5432/ai_db")

# Connexion engine.
engine = create_engine(DATABASE_URL)

# Session (tool for query).
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()