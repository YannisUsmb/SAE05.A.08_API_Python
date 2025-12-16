from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict

class RivalTrainingRequest(BaseModel):
    grid_size: int = 8
    episodes: int = 2000

class RivalTrainingResponse(BaseModel):
    message: str
    task_id: str

# State of the game at an instant T.
class GameStep(BaseModel):
    # Rover.
    step: int
    rover_pos: Tuple[int, int]
    rover_battery: int
    action_rover: str

    # Saboteur.
    saboteur_charges: Dict[str, int]
    action_saboteur: str

    # Map.
    grid_snapshot: List[List[int]]

class RivalPlayResponse(BaseModel):
    grid_size: int
    start: Tuple[int, int]
    target: Tuple[int, int]
    initial_rocks: List[Tuple[int, int]]
    history: List[GameStep]
    winner: str
    total_steps: int