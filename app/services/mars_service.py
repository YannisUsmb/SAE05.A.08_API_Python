import numpy as np
import random
import os

class MarsEnv:
    def __init__(self, size=8):
        self.size = size

        self.rover_pos = (0, 0)
        self.target = (size-1, size-1)

        self.rover_battery = 15

        self.saboteur_charges = {
            "quake": 3,
            "storm": 2
        }

        # 0 = Empty, 1 = Rock, 2 = Target.
        self.grid = np.zeros((size, size))
        self.grid[self.target] = 2

        self.cost_map = np.ones((size, size))

    def reset(self):
        self.rover_pos = (0, 0)
        self.rover_battery = 15
        self.saboteur_charges = {"quake": 3, "storm": 2}
        self.grid = np.zeros((self.size, self.size))
        self.grid[self.target] = 2
        self.cost_map = np.ones((self.size, self.size))

        # Add random rocks.
        for _ in range(self.size):
            ry, rx = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (ry, rx) not in [self.rover_pos, self.target]:
                self.grid[ry, rx] = 1

        return self._get_state()
    
    def _get_state(self):
        return self.rover_pos
    
    def step(self, action_rover, action_saboteur_type, action_saboteur_target=None):
        """
        action_rover: 0=Up, 1=Down, 2=Left, 3=Right.
        action_sentinel_type: 0=Nothing, 1=Quake, 2=Storm.
        """

        # Rover Turn.
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

        if action_saboteur_type == 1 and self.saboteur_charges["quake"] > 0:
            sy, sx = action_saboteur_target
            if (sy, sx) != self.rover_pos and (sy, sx) != self.target and self.grid[sy, sx] == 0:
                self.grid[sy, sx] = 1
                self.saboteur_charges["quake"] -= 1
                saboteur_desc = f"QUAKE @ ({sy}, {sx})"

        elif action_saboteur_type == 2 and self.saboteur_charges["storm"] > 0:
            sy, sx = action_saboteur_target
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < self.size and 0 <= nx < self.size:
                        self.cost_map[ny, nx] = 3.0
            self.saboteur_charges["storm"] -= 1
            saboteur_desc = f"STORM @ ({sy}, {sx})"

        # Verdict.

        done = False
        winner = None
        reward_rover = 0

        if self.rover_pos == self.target:
            done = True
            winner = "Rover"
            reward_rover = 100
        elif self.rover_battery <= 0:
            done = True
            winner = "Saboteur"
            reward_rover = -100
        else:
            dist = abs(self.rover_pos[0] - self.target[0]) + abs(self.rover_pos[1] - self.target[1])
            reward_rover = -0.1 * dist

        return self._get_state(), reward_rover, 0, done, winner, rover_desc, saboteur_desc
    
class MarsService:
    def __init__(self):
        self.base_path = "/app/shared_models"

    def play_match(self, grid_size=8):
        env = MarsEnv(size=grid_size)
        try:
            q_rover = np.load(f"{self.base_path}/q_rover_v2_{grid_size}.npy")
        except:
            return {"error": "AI Models not trained yet."}
        
        state = env.reset()
        done = False
        history = []
        steps = 0

        while not done and steps < 50:
            ry, rx = state
            action_r = np.argmax(q_rover[ry, rx])

            action_s_type = 0
            action_s_target = (0,0)
            if random.random() < 0.3: 
                if env.saboteur_charges["quake"] > 0:
                    action_s_type = 1
                    target_dir_y = env.target[0] - ry
                    target_dir_x = env.target[1] - rx
                    action_s_target = (ry + np.sign(target_dir_y), rx + np.sign(target_dir_x))
                elif env.saboteur_charges["storm"] > 0:
                    action_s_type = 2
                    action_s_target = env.rover_pos

            new_state, _, _, done, winner, r_desc, s_desc = env.step(action_r, action_s_type, action_s_target)
            
            history.append({
                "step": steps,
                "rover_pos": env.rover_pos,
                "rover_battery": env.rover_battery,
                "saboteur_charges": env.saboteur_charges.copy(),
                "action_rover": r_desc,
                "action_saboteur": s_desc,
                "grid_snapshot": env.grid.tolist()
            })
            state = new_state
            steps += 1
            
        return {
            "grid_size": grid_size,
            "start": (0,0),
            "target": env.target,
            "initial_rocks": [],
            "history": history,
            "winner": winner if winner else "Draw",
            "total_steps": steps
        }