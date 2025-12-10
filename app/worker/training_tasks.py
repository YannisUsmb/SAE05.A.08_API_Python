from datetime import datetime
import os
import pandas as pd
import lightgbm as lgb
import torch
import timm
from sqlalchemy import create_engine
from app.worker.celery_app import celery_app

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")

SOURCE_DB_URL = os.getenv("SOURCE_DATABASE_URL")

@celery_app.task(name="train_asteroid_model")
def train_asteroid_model_task():
    """
    Trains LightGBM using data from 't_e_asteroid_ast'.
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
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"asteroid__prediction_model_{version_id}.txt"
    output_file = os.path.join(MODEL_PATH, filename)
    bst.save_model(output_file)

    print(f"Worker saved the model {version_id} successfully at {output_file}.")
    return "Training completed."

@celery_app.task(name="train_celestial-body_model")
def train_celestialbody_model_task():
    """
    Downloads a pre-trained ResNet model and saves it for the API.
    Loop over images to fine-tune weights.
    """
    print("Worker is starting image model preparation...")
    try:
        # 'asteroid', 'black hole', 'comet', 'constellation', 'galaxy', 'nebula', 'planet', 'star'.
        model = timm.create_model('resnet18', pretrained=True, num_classes=8)
        print("ResNet18 model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model")
        return "Failed."
    
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resnet_model_{version_id}.pt"
    output_file = os.path.join(MODEL_PATH, filename)
    model.eval()

    example_input = torch.rand(1, 3, 224, 224) 
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_file)

    print(f"Worker saved the image model at {output_file}.")
    return f"Image training completed. Version: {version_id}."