from typing import Dict, Any
from .calculations import GameState, Pokemon, Move
import json


def parse_gamestate(json_data: str) -> GameState:
    data = json.loads(json_data)

    # Create Pokemon instances with stat stages
    p1_data = data["p1"]
    p2_data = data["p2"]

    p1 = Pokemon(
        name=p1_data["name"],
        evs=p1_data.get("evs"),
        stat_stages=p1_data.get("stat_stages"),
    )

    p2 = Pokemon(
        name=p2_data["name"],
        evs=p2_data.get("evs"),
        stat_stages=p2_data.get("stat_stages"),
    )

    return GameState(
        p1=p1,
        p2=p2,
        action=data["action"],
        move=Move.from_name(data.get("move")) if data.get("move") else None,
        weather=data.get("weather"),
        terrain=data.get("terrain"),
    )


def execute_from_json(json_data: str) -> Dict[str, Any]:
    """Parse JSON and execute the specified action"""
    gamestate = parse_gamestate(json_data)
    return gamestate.execute_action()
