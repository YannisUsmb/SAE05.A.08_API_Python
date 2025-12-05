import os
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine
from app.worker.celery_app import celery_app

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:postgres@db:5433/ai_db")

@celery_app.task(name="train_asteroid_model")
def train_asteroid_model_task():
    """
    Trains LightGBM using REAL data from 't_e_asteroid_ast'.
    """
    print("Worker is connecting to database to fetch training data...")

    # Create a dedicated engine for the worker.
    internal_db_url = DATABASE_URL.replace("localhost", "db").replace("5433", "5432")
    engine = create_engine(internal_db_url)

    # Fetch Data.
    query = """
    SELECT 
        ast_absolutemagnitude as absolute_magnitude,
        ast_diameterminkm as diameter_min_km,
        ast_diametermaxkm as diameter_max_km,
        ast_semimajoraxis as semi_major_axis,
        ast_inclination as inclination,
        ast_ispotentiallyhazardous as target
    FROM t_e_asteroid_ast
    WHERE 
        ast_absolutemagnitude IS NOT NULL 
        AND ast_diametermaxkm IS NOT NULL
        AND ast_semimajoraxis IS NOT NULL
    """

    try:
        df = pd.read_sql(query, engine)
        print(f"Dataset loaded: {len(df)} rows.")
    except Exception as e:
        print(f"Error reading database: {e}")
        return "Training failed."
    
    if df.empty:
        print("Warning: Dataset is empty. Cannot train.")
        return "Training skipped (Empty Data)."
    
    # Prepare Data.
    y = df['target'].astype(int)
    X = df.drop(columns=['target'])

    train_data = lgb.Dataset(X, label=y)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1
    }

    # Train.
    print("Starting LightGBM training...")
    bst = lgb.train(params, train_data, num_boost_round=100)

    # Save Model.
    output_file = os.path.join(MODEL_PATH, "asteroid_model.txt")
    bst.save_model(output_file)

    print(f"Worker saved the model successfully at {output_file}.")
    return "Training completed."