import os
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine
from app.worker.celery_app import celery_app

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")

SOURCE_DB_URL = os.getenv("SOURCE_DATABASE_URL")

@celery_app.task(name="train_asteroid_model")
def train_asteroid_model_task():
    """
    Trains LightGBM using REAL data from 't_e_asteroid_ast'.
    """
    print("Worker is connecting to database to fetch training data...")

    if not SOURCE_DB_URL:
        print("Error: SOURCE_DATABASE_URL is not set in docker-compose.")
        return "Training failed (Missing Config)."
    
    print(f"Connecting to external source DB...")

    # Create a dedicated engine for the worker.
    try:
        source_engine = create_engine(SOURCE_DB_URL)
        connection = source_engine.connect()
        connection.close()
        print("Connection to external DB successful.")
    except Exception as e:
        print(f"Could not connect to external DB: {e}")
        return "Training failed (Connection Error)."

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
        df = pd.read_sql(query, source_engine)
        print(f"Dataset loaded from external source: {len(df)} rows.")
    except Exception as e:
        print(f"Error executing query: {e}")
        return "Training failed (SQL Error)."
    
    if df.empty:
        print("Warning: External dataset is empty.")
        return "Training skipped (Empty Data)."
    
    # Prepare Data.
    if df['target'].dtype == 'bool':
        y = df['target'].astype(int)
    else:
        y = df['target']

    X = df.drop(columns=['target'])

    # Train.
    print("Starting LightGBM training...")
    train_data = lgb.Dataset(X, label=y)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    
    bst = lgb.train(params, train_data, num_boost_round=100)

    # Save Model.
    output_file = os.path.join(MODEL_PATH, "asteroid_model.txt")
    bst.save_model(output_file)

    print(f"Worker saved the model successfully at {output_file}.")
    return "Training completed."