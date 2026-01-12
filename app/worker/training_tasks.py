from datetime import datetime
import os
import pandas as pd
import lightgbm as lgb
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from sqlalchemy import create_engine
from app.services.mars_service import MarsEnv
from app.models.dqn_model import RoverDQN
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
    filename = f"asteroid_prediction_model_{version_id}.txt"
    output_file = os.path.join(MODEL_PATH, filename)
    bst.save_model(output_file)

    print(f"Worker saved the model {version_id} successfully at {output_file}.")
    return "Training completed."

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

@celery_app.task(name="train_mars_agents")
def train_mars_agents_task(grid_size= 8, episodes=1000):
    print(f"Worker is training rover against saboteur on {grid_size}x{grid_size} for {episodes} epochs.")

    env = MarsEnv(size=grid_size)
    input_dim = grid_size * grid_size
    output_dim = 4 # Up, Down, Left, Right.

    # Network Initialisation.
    policy_net = RoverDQN(input_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    buffer = ReplayBuffer(10000)

    epsilon = 1.0
    min_epsilon = 0.1
    decay = 0.995
    batch_size = 64
    gamma = 0.95

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            ry, rx = state

            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            act_s_type = 0
            act_s_target = (0,0)
            if random.random() < 0.3: # 30% chance to attack.
                if env.sentinel_charges["quake"] > 0:
                    act_s_type = 1
                    ty = env.rover_pos[0] + random.randint(-1, 1)
                    tx = env.rover_pos[1] + random.randint(-1, 1)
                    act_s_target = (max(0, min(grid_size-1, ty)), max(0, min(grid_size-1, tx)))

            # Step.
            next_state, reward, _, done, _, _, _ = env.step(action, act_s_type, act_s_target)
            
            # Storage in buffer.
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Learning.
            if len(buffer) > batch_size:
                transitions = buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                b_state = torch.FloatTensor(np.array(batch_state))
                b_action = torch.LongTensor(batch_action).unsqueeze(1)
                b_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                b_next_state = torch.FloatTensor(np.array(batch_next_state))
                b_done = torch.FloatTensor(batch_done).unsqueeze(1)

                # Actual Q(s, a).
                curr_q = policy_net(b_state).gather(1, b_action)
                
                # Max futur Q(s', a').
                next_q = policy_net(b_next_state).max(1)[0].unsqueeze(1)
                expected_q = b_reward + (gamma * next_q * (1 - b_done))

                # Gradient Descent.
                loss = loss_fn(curr_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Decay Epsilon.
            epsilon = max(min_epsilon, epsilon * decay)

            if episode % 100 == 0:
                print(f"   Episode {episode} - Reward: {total_reward:.1f} - Epsilon: {epsilon:.2f}")

    # Save.
    save_path = f"/app/shared_models/dqn_rover_{grid_size}.pth"
    torch.save(policy_net.state_dict(), save_path)
    
    return f"DQN Training finished for size {grid_size}. Saved to {save_path}."