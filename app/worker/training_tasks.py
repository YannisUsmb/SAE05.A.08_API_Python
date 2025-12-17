from datetime import datetime
import os
import pandas as pd
import lightgbm as lgb
import numpy as np
import random
from sqlalchemy import create_engine
from app.worker.celery_app import celery_app
from app.services.mars_service import MarsEnv

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

@celery_app.task(name="train_mars_agents")
def train_mars_agents_task(grid_size= 8, episodes=2000):
    print(f"Worker is training rover against saboteur on {grid_size}x{grid_size} for {episodes} epochs.")

    env = MarsEnv(size=grid_size)

    # [Y, X, Action]
    q_rover = np.zeros((grid_size, grid_size, 4))

    epsilon = 1.0
    decay = 0.995
    min_epsilon = 0.05
    lr = 0.1
    gamma = 0.95

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            ry, rx = state

            if random.random() < epsilon:
                act_r = random.randint(0, 3)
            else:
                act_r = np.argmax(q_rover[ry, rx])

            act_s_type = 0
            act_s_target = (0,0)
            if random.random() < 0.2: # 20% chance to attack.
                act_s_type = random.choice([1, 2])
                ty = min(grid_size-1, max(0, ry + random.randint(-1, 1)))
                tx = min(grid_size-1, max(0, rx + random.randint(-1, 1)))
                act_s_target = (ty, tx)

            # Step.
            next_state, r_r, _, done, _, _, _ = env.step(act_r, act_s_type, act_s_target)
            nry, nrx = next_state
            
            # Update Rover Q-Table.
            best_next = np.max(q_rover[nry, nrx])
            q_rover[ry, rx, act_r] += lr * (r_r + gamma * best_next - q_rover[ry, rx, act_r])
            
            state = next_state
            
        epsilon = max(min_epsilon, epsilon * decay)
        
        if episode % 200 == 0:
            print(f"    Episode {episode}/{episodes} - Epsilon: {epsilon:.2f}")

    # Save.
    np.save(f"/app/shared_models/q_rover_{grid_size}.npy", q_rover)
    
    return f"Training finished for size {grid_size}."