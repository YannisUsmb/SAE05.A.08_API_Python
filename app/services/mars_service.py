import numpy as np
import random
import traceback
import torch
from app.models.cnn_model import RoverCNN

class MarsEnv:
    def __init__(self, size=8):
        self.size = size

        # 0 = Empty, 1 = Rock, 2 = Storm, 3 = Base.

        self.rover_pos = (0, 0)
        self.target = (size-1, size-1)

        self.rover_battery = 15

        self.saboteur_charges = {
            "quake": 3,
            "storm": 2
        }

        self.grid = np.zeros((size, size))
        self.grid[self.target] = 3

        self.cost_map = np.ones((size, size))

        self.visited = set()

    def reset(self):
        self.rover_battery = 15
        self.saboteur_charges = {"quake": 3, "storm": 1}
        self.visited = set()

        # Random Generation of start and target.
        while True:
            y1, x1 = random.randint(0, self.size-1), random.randint(0, self.size-1)
            y2, x2 = random.randint(0, self.size-1), random.randint(0, self.size-1)

            # Check that they are not in the same place.
            dist = abs(y1-y2) + abs(x1-x2)
            if (y1, x1) != (y2, x2) and dist > 3:
                self.rover_pos = (y1, x1)
                self.target = (y2, x2)
                break

        self.visited.add(self.rover_pos)
        
        self.grid = np.zeros((self.size, self.size))
        self.cost_map = np.ones((self.size, self.size)) 

        # Add random rocks (never on the Rover or the Base).
        for _ in range(self.size):
            ry, rx = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (ry, rx) not in [self.rover_pos, self.target]:
                self.grid[ry, rx] = 1

        return self._get_state()
    
    def _get_state(self):
        # Canal 0: Rover Position.
        channel_rover = np.zeros((self.size, self.size), dtype=np.float32)
        if 0 <= self.rover_pos[0] < self.size and 0 <= self.rover_pos[1] < self.size:
            channel_rover[self.rover_pos] = 1.0

        # Canal 1: Obstacles.
        channel_obstacles = np.zeros((self.size, self.size), dtype=np.float32)
        channel_obstacles[self.grid == 1] = 1.0 # Walls.
        channel_obstacles[self.grid == 2] = 0.5 # Storms.

        # Canal 2: Base.
        channel_target = np.zeros((self.size, self.size), dtype=np.float32)
        channel_target[self.target] = 1.0

        # Stack the 3 channels: Shape (3, 8, 8).
        state = np.stack([channel_rover, channel_obstacles, channel_target], axis=0)
        return state
    
    def step(self, action_rover, action_saboteur_type, action_saboteur_target=None):
        """
        action_rover: 0=Up, 1=Down, 2=Left, 3=Right.
        action_sentinel_type: 0=Nothing, 1=Quake, 2=Storm.
        """

        # Rover Turn.
        old_pos = self.rover_pos

        dy, dx = 0, 0
        rover_desc = ["UP", "DOWN", "LEFT", "RIGHT"][action_rover]

        if action_rover == 0: dy = -1
        elif action_rover == 1: dy = 1
        elif action_rover == 2: dx = -1
        elif action_rover == 3: dx = 1

        ny, nx = self.rover_pos[0] + dy, self.rover_pos[1] + dx

        if 0 <= ny < self.size and 0 <= nx < self.size and self.grid[ny, nx] != 1:
            self.rover_pos = (ny, nx)

        cost = self.cost_map[self.rover_pos]
        self.rover_battery -= cost

        # Saboteur Turn.
        saboteur_desc = "WAIT"

        # Quake Action (Rock=1).
        if action_saboteur_type == 1 and self.saboteur_charges["quake"] > 0:
            sy, sx = action_saboteur_target
            if (sy, sx) != self.rover_pos and (sy, sx) != self.target and self.grid[sy, sx] == 0:
                self.grid[sy, sx] = 1
                self.saboteur_charges["quake"] -= 1
                saboteur_desc = f"QUAKE @ ({sy}, {sx})"

        # Storm Action (Storm=2).
        elif action_saboteur_type == 2 and self.saboteur_charges["storm"] > 0:
            sy, sx = action_saboteur_target
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < self.size and 0 <= nx < self.size:
                        self.cost_map[ny, nx] = 3.0

                        if self.grid[ny, nx] == 0:
                            self.grid[ny, nx] = 2

            self.saboteur_charges["storm"] -= 1
            saboteur_desc = f"STORM @ ({sy}, {sx})"

        # Verdict.
        done = False
        winner = None
        reward_rover = 0

        dist = abs(self.rover_pos[0] - self.target[0]) + abs(self.rover_pos[1] - self.target[1])

        if self.rover_pos == self.target:
            done = True
            winner = "Rover"
            reward_rover = 100
        elif self.rover_battery <= 0:
            done = True
            winner = "Saboteur"
            reward_rover = -100
        else:
            # Reward Shaping (closer is better).
            reward_rover = (15.0 - dist) * 0.1 - 0.5

            # Wall Penalty.
            if self.rover_pos == old_pos:
                reward_rover -= 5.0

            # Visited (encourage exploration).
            if self.rover_pos not in self.visited:
                reward_rover += 0.5 # Bonus for discovering.
                self.visited.add(self.rover_pos)
            else:
                reward_rover -= 0.2 # Penalty if return.

        return self._get_state(), reward_rover, 0, done, winner, rover_desc, saboteur_desc
    
class MarsService:
    def __init__(self):
        self.base_path = "/app/shared_models"

    def play_match(self, grid_size=8):
        try:
            env = MarsEnv(size=grid_size)

            # Load architecture.
            model = RoverCNN(4)

            # Load weights.
            path = f"{self.base_path}/cnn_rover_{grid_size}.pth"
            print(f"Loading DQN from {path}")

            try:
                model.load_state_dict(torch.load(path))
                model.eval()
            except Exception as e:
                return {"error": f"Model error: {e}", "history": [], "grid_size": grid_size, "total_steps": 0}
            
            state = env.reset()
            done = False
            history = []
            steps = 0

            while not done and steps < 50:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)
                    action_r = q_values.argmax().item()

                ry, rx = int(env.rover_pos[0]), int(env.rover_pos[1])

                action_s_type = 0
                action_s_target = (0,0)

                if random.random() < 0.4: 
                    if env.saboteur_charges["quake"] > 0:
                        action_s_type = 1

                        best_block_pos = None
                        min_dist_to_target = 999

                        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                            ny, nx = ry + dy, rx + dx
                            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                                # If it's empty and not the base.
                                if env.grid[ny, nx] == 0 and (ny, nx) != env.target:
                                    d = abs(ny - env.target[0]) + abs(nx - env.target[1])
                                    if d < min_dist_to_target:
                                        min_dist_to_target = d
                                        best_block_pos = (ny, nx)

                        if best_block_pos:
                            action_s_target = best_block_pos
                        else:
                            action_s_target = (ry, rx)

                    elif env.saboteur_charges["storm"] > 0:
                        action_s_type = 2
                        action_s_target = (ry, rx)

                new_state, _, _, done, winner, r_desc, s_desc = env.step(action_r, action_s_type, action_s_target)

                display_grid = env.grid.copy()
                display_grid[env.target] = 3

                step_data = {
                    "step": int(steps),
                    "rover_pos": (ry, rx),
                    "rover_battery": float(env.rover_battery),
                    "saboteur_charges": {k: int(v) for k, v in env.saboteur_charges.items()},
                    "action_rover": str(r_desc),
                    "action_saboteur": str(s_desc),
                    "grid_snapshot": [[int(cell) for cell in row] for row in env.grid]
                }
                history.append(step_data)

                state = new_state
                steps += 1
                
            return {
                "grid_size": int(grid_size),
                "start": (0,0),
                "target": (int(env.target[0]), int(env.target[1])),
                "initial_rocks": [],
                "history": history,
                "winner": str(winner) if winner else "Draw",
                "total_steps": int(steps)
            }
        
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"API Crash: {error_msg}")
            return {
                "error": f"Internal Crash: {str(e)}",
                "history": [], "grid_size": 0, "start": (0,0), "target": (0,0), "initial_rocks": [], "total_steps": 0
            }