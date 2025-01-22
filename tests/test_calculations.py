import pytest
from app.utils.calculations import calc_stat, stat_modifier, poke_round
from app.utils.calculations import Pokemon, Move, GameState, read_in_pokemon


def test_calc_stat_hp():
    quaxly_hp = calc_stat(50, 55, 252, 31, True)

    assert quaxly_hp == 162, "HP calc is wrong"


def test_calc_stat_atk():
    quaxly_atk = calc_stat(50, 65, 252, 31, False)

    assert quaxly_atk == 117, "Atk calc is wrong"


def test_stat_modifier_pos():

    #  gholdengo stat checks
    ghol_test_1 = stat_modifier(num_stages=2, stat=calc_stat(50, 84, 252, 31))
    assert ghol_test_1 == 272, "Ghol test is wrong"


def test_stat_modifier_neg():

    ghol_test_2 = stat_modifier(num_stages=-3, stat=calc_stat(50, 84, 252, 31))
    assert ghol_test_2 == 54, "Ghol test is wrong"


def test_poke_round():
    # classic test written in article from DeWoblefet on damage calculation

    assert poke_round(30.2) == 30, "poke_round is wrong"
    assert poke_round(30.5) == 30, "poke_round is wrong"
    assert poke_round(30.7) == 31, "poke_round is wrong"


# read in pokemon data as pytest fixture


@pytest.fixture
def pokemon_data():
    pokemons = read_in_pokemon("./data/gen9_pokemon.jsonl")
    return pokemons


@pytest.fixture
def fuecoco():
    fuecoco_evs = {
        "hp": 0,
        "attack": 252,
        "defense": 0,
        "special-attack": 0,
        "special-defense": 0,
        "speed": 252,
    }
    pokemon = Pokemon(name="Fuecoco", evs=fuecoco_evs, stat_stages={"speed": 2})
    return pokemon


@pytest.fixture
def sprigatito():
    sprigatito_evs = {
        "hp": 0,
        "attack": 252,
        "defense": 0,
        "special-attack": 0,
        "special-defense": 0,
        "speed": 252,
    }
    return Pokemon(name="Sprigatito", evs=sprigatito_evs)


@pytest.fixture
def charizard():
    charizard_evs = {
        "hp": 0,
        "attack": 252,
        "defense": 0,
        "special-attack": 252,
        "special-defense": 0,
        "speed": 0,
    }
    return Pokemon(name="Charizard", evs=charizard_evs)


def test_nature_modifier_increase():
    # test that the nature meodi

    # special attack increase
    pokemon = Pokemon(name="Charizard", nature="quiet")

    # check that special attack stat is 141
    assert (
        pokemon.trained_stats["special-attack"] == 141
    ), "Nature modifier isn't increasing correctly"


def test_nature_modifier_decrease():
    # tests that nature stat is decreased as expected

    # speed decrease

    pokemon = Pokemon(name="Charizard", nature="quiet")
    assert (
        pokemon.trained_stats["speed"] == 108
    ), "Nature modifier isn't decreasing correctly"
    pass


def test_calculate_base_damage_sprigatito(sprigatito, fuecoco):
    game_state = GameState(
        p1=sprigatito, p2=fuecoco, action="attack", move=Move.from_name("Tackle")
    )
    result = game_state.execute_action()
    assert result["min_damage"] == 22
    assert result["max_damage"] == 27


def test_calculate_damage_sun_fire(charizard, sprigatito):
    game_state = GameState(
        p1=charizard,
        p2=sprigatito,
        action="attack",
        move=Move.from_name("Fire Blast"),
        weather="sun",
    )
    result = game_state.execute_action()
    assert result["min_damage"] == 458
    assert result["max_damage"] == 542


# write test for same scenario, but rain instead of sun
def test_calculate_base_damage_rain_fire(charizard, sprigatito):
    game_state = GameState(
        p1=charizard,
        p2=sprigatito,
        action="attack",
        move=Move.from_name("Fire Blast"),
        weather="rain",
    )
    result = game_state.calculate_modified_damage()
    assert (
        result["min_damage"] == 152 and result["max_damage"] == 180
    ), "Base damage calculation is wrong for Charizard in Rain with Fire Blast"


def test_calculate_damage_fully_loaded(charizard, sprigatito):
    # mean to activate as many modifiers as possible
    game_state = GameState(
        p1=charizard,
        p2=sprigatito,
        action="attack",
        move=Move.from_name("Fire Blast"),
        weather="rain",
    )

    # charizard modifiers
    charizard.stat_stages = {"special-attack": 2}
    charizard.tera_active = True
    charizard.tera_type = "fire"
    charizard.item = "specs"

    # sprigatito modifiers
    sprigatito.stat_stages = {"special-defense": 1}
    sprigatito.tera_active = True
    sprigatito.tera_type = "water"
    sprigatito.item = "av"

    result = game_state.calculate_modified_damage(verbose=True)

    assert (
        result["min_damage"] == 68 and result["max_damage"] == 81
    ), "Base damage calculation is wrong for Charizard in Rain with Fire Blast"


# Spread Moves


# Critical Hit


@pytest.fixture
def finneon():
    return Pokemon(name="finneon", evs={"speed": 230})


@pytest.fixture
def meowth():
    return Pokemon(name="meowth", evs={"speed": 215})


@pytest.fixture
def salamence():
    return Pokemon(name="salamence", evs={"speed": 252})


@pytest.fixture
def talonflame():
    return Pokemon(name="talonflame", evs={"speed": 248})


@pytest.fixture
def iron_bundle():
    return Pokemon(name="iron-bundle", evs={"speed": 252})


@pytest.fixture
def flutter_mane():
    return Pokemon(name="flutter-mane", evs={"speed": 252})


@pytest.fixture
def quaxly():
    return Pokemon(name="quaxly", evs={"speed": 4}, stat_stages={"speed": 2})


def test_speed_check_finneon_meowth(finneon, meowth):
    game_state = GameState(p1=finneon, p2=meowth, action="speed_check")
    result = game_state._check_speed()
    assert result["p1_final_speed"] == 115
    assert result["p2_final_speed"] == 137
    assert result["result"] == "p2_faster"


def test_speed_check_salamence_talonflame(salamence, talonflame):
    game_state = GameState(p1=salamence, p2=talonflame, action="speed_check")
    result = game_state._check_speed()
    assert result["p1_final_speed"] == 152
    assert result["p2_final_speed"] == 177
    assert result["result"] == "p2_faster"


def test_speed_check_iron_bundle_flutter_mane(iron_bundle, flutter_mane):
    game_state = GameState(p1=iron_bundle, p2=flutter_mane, action="speed_check")
    result = game_state._check_speed()
    assert result["p1_final_speed"] == 188
    assert result["p2_final_speed"] == 187
    assert result["result"] == "p1_faster"


def test_speed_check_boosted_fuecoco_quaxly(fuecoco, quaxly):
    game_state = GameState(p1=fuecoco, p2=quaxly, action="speed_check")
    result = game_state._check_speed()
    assert result["p1_final_speed"] == 176
    assert result["p2_final_speed"] == 142
    assert result["result"] == "p1_faster"
    assert result["p1_stat_changes"] == 2
    assert result["p2_stat_changes"] == 2
