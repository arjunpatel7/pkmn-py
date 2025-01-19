import pytest
from app.utils.calculations import Pokemon, Move, GameState, calculate_optimal_evs
import pandas as pd

# This will will test the train_pokemon script

# Test a game state with a pokemon with 252 EVs in HP and the corresponding defensive stat


@pytest.fixture
def charizard():
    charizard_evs = {
        "hp": 0,
        "attack": 0,
        "defense": 0,
        "special-attack": 252,
        "special-defense": 0,
        "speed": 0,
    }
    return Pokemon(name="Charizard", evs=charizard_evs, nature=None)


@pytest.fixture
def eevee():
    eevee_evs = {
        "hp": 0,
        "attack": 0,
        "defense": 0,
        "special-attack": 0,
        "special-defense": 0,
        "speed": 0,
    }
    return Pokemon(name="Eevee", evs=eevee_evs, nature=None)


def test_optimal_ev_ohko(charizard, eevee):
    # test ohko
    move = Move(name="Overheat", type="fire", category="special", base_power=130)
    practice_gamestate = GameState(p1=charizard, p2=eevee, move=move)
    results = pd.DataFrame(calculate_optimal_evs(practice_gamestate))
    assert results[results["training"] == "optimal"].evs_invested.iloc[0] == 172
