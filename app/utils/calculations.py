import math
import editdistance
import jsonlines
from typing import Dict, Optional, List, Any, ClassVar
from .consts import (
    offensive_type_resistance,
    offensive_type_effectiveness,
    offensive_type_immunities,
    natures,
)
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json

# class based implementation of pokemon calculator
# TODO: Add in terrain modifications
# TODO: Add in nature modifications

MOVE_DATA_PATH = "./data/move_data.jsonl"


class Pokemon(BaseModel):
    name: str
    evs: Optional[Dict[str, int]] = None
    nature: Optional[str] = None
    tera_type: Optional[str] = None
    tera_active: bool = False
    status: Optional[str] = None
    DEFAULT_STAT_STAGES: ClassVar[Dict[str, int]] = {
        "attack": 0,
        "defense": 0,
        "special-attack": 0,
        "special-defense": 0,
        "speed": 0,
    }
    # updated specified stats only
    stat_stages: Dict[str, int] = Field(
        default_factory=lambda: Pokemon.DEFAULT_STAT_STAGES.copy()
    )
    item: Optional[str] = None

    # Computed fields
    types: List[str] = []
    stats: Dict[str, int] = {}
    trained_stats: Dict[str, int] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("evs")
    def validate_evs(cls, v):
        if v is None:
            return create_empty_ev_spread()
        for stat, value in v.items():
            if not 0 <= value <= 252:
                raise ValueError(f"EV for {stat} must be between 0 and 252")
        return v

    @field_validator("nature")
    def validate_nature(cls, v):
        if v is not None and v not in natures:
            raise ValueError(f"Invalid nature: {v}")
        return v

    def __init__(self, **data):
        # If stat_stages is provided, merge it with defaults instead of replacing
        if "stat_stages" in data:
            partial_stages = data["stat_stages"]
            full_stages = Pokemon.DEFAULT_STAT_STAGES.copy()
            full_stages.update(partial_stages)
            data["stat_stages"] = full_stages

        super().__init__(**data)
        pokemon = lookup_pokemon(
            self.name, read_in_pokemon("./data/gen9_pokemon.jsonl")
        )
        if pokemon:
            # Set types
            types = pokemon["types"]
            self.types = [types] if isinstance(types, str) else types
            # Set stats
            self.stats = {x["stat"]["name"]: x["base_stat"] for x in pokemon["stats"]}
            # Set trained stats
            self.trained_stats = create_trained_stats(self.evs, self.stats, self.nature)

    def stat_stage_increase(self, stat: str, num_stages: int):
        # when a pokemon's stat increases, modify the stat and the stage_stages

        current_stages = self.stat_stages[stat]
        if current_stages is None:
            # if no stat stages, then just set it to 0
            current_stages[stat] = 0
        self.stat_stages[stat] = current_stages + num_stages
        self.trained_stats[stat] = stat_modifier(num_stages, self.trained_stats[stat])

    def retrain(self, stat: str, ev: int):
        # retrain a pokemon with new evs

        if self.evs is None:
            self.evs = create_empty_ev_spread()

        self.evs[stat] = ev
        # upgrade stats based on evs
        self.trained_stats[stat] = calc_stat(50, self.stats[stat], self.evs[stat], 31)
        return self

    def pretty_print(self):
        # prints out the pokemon name, and all stats, and attributes

        print(f"Pokemon is {self.name}")
        print(f"Pokemon types are {self.types}")
        print(f"Pokemon stats are {self.stats}")
        print(f"Pokemon evs are {self.evs}")
        print(f"Pokemon nature is {self.nature}")
        print(f"Pokemon tera type is {self.tera_type}")
        print(f"Pokemon tera active is {self.tera_active}")
        print(f"Pokemon status is {self.status}")
        print(f"Pokemon stat stages are {self.stat_stages}")

        # fill in stat distributions


class Move(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    category: Optional[str] = None
    base_power: Optional[int] = Field(default=None, ge=0)
    priority: int = 0
    description: Optional[str] = None

    @classmethod
    def from_name(cls, name: str):
        # Convert spaces to hyphens and make lowercase
        formatted_name = name.lower().replace(" ", "-")

        # Read from move_data.jsonl
        with open("data/move_data.jsonl", "r") as f:
            for line in f:
                move = json.loads(line)
                if move["name"] == formatted_name:  # Compare with formatted name
                    return cls(
                        name=move["name"],
                        base_power=move[
                            "power"
                        ],  # Note: this can be None for some moves
                        category=move["category"],
                        type=move["type"],
                    )

        raise ValueError(
            f"Move {name} (formatted as '{formatted_name}') not found in move database"
        )


class GameState(BaseModel):
    p1: Pokemon
    p2: Pokemon
    move: Optional[Move] = None
    action: str = "attack"
    weather: Optional[str] = None
    terrain: Optional[str] = None
    critical_hit: bool = False
    action_params: Optional[Dict[str, Any]] = None

    @field_validator("weather")
    def validate_weather(cls, v):
        valid_weather = ["sun", "rain", "sand", "hail", None]
        if v not in valid_weather:
            raise ValueError(f"Invalid weather: {v}")
        return v

    @field_validator("terrain")
    def validate_terrain(cls, v):
        valid_terrain = ["electric", "grassy", "misty", "psychic", None]
        if v not in valid_terrain:
            raise ValueError(f"Invalid terrain: {v}")
        return v

    @field_validator("action")
    def validate_action(cls, v):
        valid_actions = ["attack", "speed_check", "train"]
        if v not in valid_actions:
            raise ValueError(f"Invalid action: {v}")
        return v

    def _check_speed(self) -> dict:
        """Perform speed comparison between p1 and p2"""
        p1_speed = self.p1.trained_stats["speed"]
        p2_speed = self.p2.trained_stats["speed"]

        # Apply stat stage changes if they exist
        attacking_stat_changes = self.p1.stat_stages["speed"]
        defending_stat_changes = self.p2.stat_stages["speed"]

        p1_final_speed = stat_modifier(num_stages=attacking_stat_changes, stat=p1_speed)
        p2_final_speed = stat_modifier(num_stages=defending_stat_changes, stat=p2_speed)

        result = {
            "p1": self.p1.name,
            "p2": self.p2.name,
            "p1_final_speed": p1_final_speed,
            "p2_final_speed": p2_final_speed,
            "p1_stat_changes": attacking_stat_changes,
            "p2_stat_changes": defending_stat_changes,
            "p1_ev": self.p1.evs["speed"] if self.p1.evs else 0,
            "p2_ev": self.p2.evs["speed"] if self.p2.evs else 0,
        }

        # Add who's faster to the result
        if p1_final_speed == p2_final_speed:
            result["result"] = "speed_tie"
            result["message"] = f"Speed Tie, with both pokemon at {p1_final_speed}"
        elif p1_final_speed > p2_final_speed:
            result["result"] = "p1_faster"
            result[
                "message"
            ] = f"{self.p1.name} speed stat is {p1_final_speed}, which is faster than {self.p2.name} at {p2_final_speed}"
        else:
            result["result"] = "p2_faster"
            result[
                "message"
            ] = f"{self.p1.name} speed stat is {p1_final_speed}, which is slower than {self.p2.name} at {p2_final_speed}"

        return result

    def execute_action(self) -> dict:
        """Route to appropriate calculation based on action"""
        action_map = {
            "attack": self.calculate_modified_damage,
            "speed_check": self._check_speed,
            "train": lambda: calculate_optimal_evs(
                self, self.action_params.get("criteria", {})
            ),
        }

        if self.action not in action_map:
            raise ValueError(f"Unknown action: {self.action}")

        return action_map[self.action]()

    def calculate_base_damage(self):
        # given args, calculatse base damage of attack

        # lookup if move is physical or special

        category = self.move.category
        attacking_stat = "attack" if category == "physical" else "special-attack"
        defending_stat = "defense" if category == "physical" else "special-defense"

        # pull stat changes and stats from pokemon
        attacking_stat_changes = self.p1.stat_stages[attacking_stat]
        defending_stat_changes = self.p2.stat_stages[defending_stat]

        attacking_stat = stat_modifier(
            num_stages=attacking_stat_changes,
            stat=self.p1.trained_stats[attacking_stat],
        )
        defending_stat = stat_modifier(
            num_stages=defending_stat_changes,
            stat=self.p2.trained_stats[defending_stat],
        )

        LEVEL = 50

        level_weight = math.floor(((2 * LEVEL) / 5) + 2)

        # modify move base power based on item
        if self.p1.item is not None:
            self.move.base_power = bp_item_modifier(self.move, self.p1.item)

        step1 = math.floor(
            (level_weight * self.move.base_power * attacking_stat) / defending_stat
        )

        step2 = math.floor(step1 / 50) + 2

        return step2

    def calculate_modified_damage(self, verbose=False):
        base_damage = self.calculate_base_damage()
        verbose_print(verbose, message="Base damage of move is", result=base_damage)

        # spread move modifier
        final_damage = spread_move_modifier(base_damage)
        verbose_print(
            verbose, message="spread modified damage of move is", result=final_damage
        )

        # weather modifier
        final_damage = weather_modifier(self.weather, self.move.type, final_damage)
        verbose_print(
            verbose, message="weather modified damage of move is:", result=final_damage
        )

        # critical hit modifier
        final_damage = critical_hit_modifier(final_damage, self.critical_hit)
        verbose_print(
            verbose,
            message="critical hit modified damage of move is",
            result=final_damage,
        )
        # random modifier
        final_damage_min, final_damage_max = random_modifier(final_damage)
        verbose_print(verbose, message="random min of move is", result=final_damage_min)
        verbose_print(verbose, message="random max of move is", result=final_damage_max)

        # tera effectiveness modifier
        final_damage_min = tera_modifier(self.p1, self.move.type, final_damage_min)
        verbose_print(
            verbose, message="tera and stab effectiveness", result=final_damage_min
        )
        final_damage_max = tera_modifier(self.p1, self.move.type, final_damage_max)
        verbose_print(
            verbose, message="min damage after tera/stab mod", result=final_damage_min
        )

        verbose_print(
            verbose, message="max damage after tera/stab mod", result=final_damage_max
        )

        # type effectiveness modifier
        final_damage_min = type_modifier(final_damage_min, self)
        final_damage_max = type_modifier(final_damage_max, self)
        verbose_print(
            verbose,
            message="min damage after type effectiveness",
            result=final_damage_min,
        )
        verbose_print(
            verbose,
            message="max damage after type effectiveness",
            result=final_damage_max,
        )

        # burn modifier
        final_damage_min = burn_modifier(
            final_damage_min, self.move.category, self.p1.status
        )
        final_damage_max = burn_modifier(
            final_damage_max, self.move.category, self.p1.status
        )

        # final modifier/special cases

        return {
            "min_damage": int(final_damage_min),  # Ensure integers
            "max_damage": int(final_damage_max),
        }


def verbose_print(verbose, result, message=""):
    if verbose:
        print(message, result)


def poke_round(num):
    # given a number, round it down if decimal is less than 0.5
    # round up if decimal is greater than 0.5

    decimal = num - math.floor(num)
    return math.floor(num) if decimal <= 0.5 else math.ceil(num)


def spread_move_modifier(damage, is_spread=False):
    # reduce damage by 0.75 and pokeround
    if is_spread:
        return poke_round(damage * (3072 / 4096))
    return damage


def read_in_moves(f):
    moves = []
    with jsonlines.open(f) as reader:
        for entry in reader:
            moves.append(entry)
    return moves


def lookup_move(move_name, move_data_path):

    # read in move data
    moves = read_in_moves(move_data_path)
    selected_move = None
    for move in moves:
        if move["name"] == move_name:
            selected_move = move
            break
    if selected_move is not None:
        move = Move(**selected_move)
    return None


def random_modifier(damage):
    # returns two values, which are the min and max damage possible

    return math.floor((damage * 85) / 100), poke_round(damage)


def STAB_modifier(damage):
    return poke_round((6144 / 4096) * damage)


def tera_modifier(pokemon, move_type, damage):
    # tera typing where the move is same type as pokemon
    # just doubles the stab modifier
    # otherwise if the types are different, we do a normal stab bonus
    pokemon_types = pokemon.types
    tera_type = pokemon.tera_type
    tera_active = pokemon.tera_active

    if tera_active:
        if tera_type == move_type:
            # then just do the 1.5x modifier
            if tera_type in pokemon_types:
                # then do the 2x modifier
                return poke_round((8192 / 4096) * damage)
            return STAB_modifier(damage)
    # then no modifier
    elif move_type in pokemon_types:
        return STAB_modifier(damage)
    else:
        return damage


def item_modifier(pokemon, item_class):
    if item_class in ["band", "banded", "choice band"]:
        # choice band ups attack by 1.5x
        pokemon.stat["attack"] = poke_round((6144 / 4096) * pokemon.stat["attack"])
    elif item_class in ["specs", "choice specs"]:
        # choice specs ups special attack by 1.5x
        pokemon.stat["special_attack"] = poke_round(
            (6144 / 4096) * pokemon.stat["special_attack"]
        )
    elif item_class in ["scarf", "choice scarf"]:
        # choice scarf ups speed by 1.5x
        pokemon.stat["speed"] = poke_round((6144 / 4096) * pokemon.stat["speed"])
    elif item_class in ["assault vest", "vest", "av"]:
        # assault vest ups special defense by 1.5x
        pokemon.stat["special_defense"] = poke_round(
            (6144 / 4096) * pokemon.stat["special_defense"]
        )
    return pokemon


def bp_item_modifier(move, item):
    # base power modifiers that occur with special items
    # for now, these are using placeholders to refer to classes of items
    if item == "boosted":
        # generic 1.2x boost
        return poke_round((12288 / 4096) * move.base_power)
    elif item == "life orb":
        # life orb 1.3x boost
        return poke_round((13312 / 4096) * move.base_power)
    return move.base_power


def weather_modifier(weather, move_type, damage):
    # account for sun or rain only
    # boosts for rain and water, and sun and fire
    # halves for rain and fire, and sun and water

    # does not do defense boost for hail or special
    # defense for sandstorm

    if weather == "rain" and move_type == "water":
        return poke_round((6144 / 4096) * damage)
    elif weather == "sun" and move_type == "fire":
        return poke_round((6144 / 4096) * damage)
    elif weather == "rain" and move_type == "fire":
        return poke_round((2048 / 4096) * damage)
    elif weather == "sun" and move_type == "water":
        return poke_round((2048 / 4096) * damage)
    else:
        return damage


def critical_hit_modifier(damage, critical_hit=False):
    # need to double check this one
    if critical_hit:
        return poke_round((6144 / 4096) * damage)
    return damage


# BST to actual stat


def calc_stat(level, base, ev, iv, is_hp=False):
    first_term = math.floor((((2 * base) + iv + math.floor(ev / 4)) * level) / 100)
    if is_hp:
        return level + first_term + 10
    else:
        return math.floor(first_term + 5)


# Attack stat modifirs


def stat_modifier(num_stages, stat):
    # given stat change, compute modifier and new stat
    modifier = 1
    direction = 1
    if num_stages == 0:
        return stat
    if num_stages < 0:
        direction = -1
        num_stages = abs(num_stages)
    if num_stages >= 6:
        modifier = 4
    elif num_stages == 5:
        modifier = 7 / 2
    elif num_stages == 4:
        modifier = 3
    elif num_stages == 3:
        modifier = 5 / 2
    elif num_stages == 2:
        modifier = 2
    elif num_stages == 1:
        modifier = 3 / 2

    if direction == -1:
        return math.floor((1 / modifier) * stat)
    return math.floor(modifier * stat)


def immunity_check(p2_type, move_type):
    # first, we need to check that the move type is in the immunities dictionary
    # then we need to check if the pokemon has the type that is immune to the move
    if move_type in offensive_type_immunities:
        return p2_type in offensive_type_immunities[move_type]
    return False


def type_mulitplier_lookup(p2_type, move_type):
    # returns mulitplier for type effectiveness and resistance in a list
    is_resisted = p2_type in offensive_type_resistance[move_type]
    is_effective = p2_type in offensive_type_effectiveness[move_type]

    is_immune = immunity_check(p2_type, move_type)

    if is_immune:
        return 0
    if is_resisted:
        return 0.5
    elif is_effective:
        return 2
    elif is_immune:
        return 0
    else:
        return 1


def type_multiplier(p2_type, move_type):
    # returns mulitplier for type effectiveness and resistance
    modifiers = [type_mulitplier_lookup(x, move_type) for x in p2_type]
    # good for if there is only one type
    return math.prod(modifiers)


def type_modifier(damage, gamestate):
    # grab types
    p2_type = gamestate.p2.types
    # grab types of move
    move_type = gamestate.move.type
    # override types if tera typing is active
    if gamestate.p2.tera_active:
        p2_type = [gamestate.p2.tera_type]

    # calculate type modifier
    return damage * type_multiplier(p2_type, move_type)

    # return type modifier


def burn_modifier(damage, category, status):
    # if pokemon is burned, and category is physical, then damage is halved
    if status == "burn" and category == "physical":
        return poke_round((2048 / 4096) * damage)
    return damage


def nature_modifier(stats, nature):
    # given stats and nature, return modified stats
    # nature is a dictionary of stat to multiplier
    if nature is not None:
        for stat, mod in natures[nature].items():
            stats[stat] = math.floor(stats[stat] * mod)
    return stats


def create_empty_ev_spread():
    # create an empty ev spread
    return {
        x: 0
        for x in [
            "hp",
            "attack",
            "defense",
            "special-attack",
            "special-defense",
            "speed",
        ]
    }


def create_trained_stats(evs, stats, nature=None):
    # given a pokemon's evs and stats, return trained stats
    # handles no-evs case by setting evs to 0
    if evs is None:
        evs = create_empty_ev_spread()
    # if not all evs are present, then fill in the rest with 0
    if len(evs) < 6:
        ev_spread = evs.keys()
        for stat in stats:
            if stat not in ev_spread:
                evs[stat] = 0
    ev_spread = evs.keys()
    trained_stats = {}
    for stat in stats:
        if stat == "hp":
            trained_stats[stat] = calc_stat(50, stats[stat], evs[stat], 31, is_hp=True)
        elif stat in ev_spread:
            trained_stats[stat] = calc_stat(50, stats[stat], evs[stat], 31)
        else:
            trained_stats[stat] = calc_stat(50, stats[stat], 0, 31)

    trained_stats = nature_modifier(trained_stats, nature)
    return trained_stats


def read_in_pokemon(f):
    pokemons = []
    with jsonlines.open(f) as reader:
        for entry in reader:
            pokemons.append(entry)
    return pokemons


def lookup_pokemon(pokemon, pokemons):
    # given pokemon name, return pokemon dict of stats

    # first, check for exact match
    # if no exact match,check for edit distance closest

    # preprocess pokemon names to avoid errors
    # check for whitespace, lowercase, commas, everything except hypens
    # and apostrophes

    # preprocess pokemon name
    pokemon = pokemon.replace(" ", "").replace(",", "").lower()

    all_pokemon = [x["name"] for x in pokemons]

    matched_pokemon = pokemon if pokemon in all_pokemon else None
    if matched_pokemon is None:
        # find the closest name by edit distance
        matched_pokemon = min(
            all_pokemon, key=lambda x: abs(editdistance.eval(pokemon, x))
        )
        # add a condition here that if the edit distance is too large
        # then we should return None
        if editdistance.eval(pokemon, matched_pokemon) > 4:
            return None

    # given pokemon dict, just grab relevant mon and return
    for poke in pokemons:
        if poke["name"] == matched_pokemon:
            return poke


# this function needs to be refactored to reflect changes in data_collection.py


def extract_stat(p, stat):
    return list(filter(lambda x: x["stat"]["name"] == stat, p["stats"]))[0]["base_stat"]


def calculate_optimal_evs(game_state: GameState) -> List[Dict]:
    """
    Calculates optimal EVs for 1hko for the attacking pokemon
    """
    results = []
    criteria = (
        game_state.action_params.get("criteria", "1hko")
        if game_state.action_params
        else "1hko"
    )

    higher_stat_name = (
        "attack" if game_state.move.category == "physical" else "special-attack"
    )

    # Calculate remaining EVs
    if game_state.p1.evs:
        evs_remaining = min(508 - sum(game_state.p1.evs.values()), 252)
    else:
        evs_remaining = 252

    # Determine minimum damage needed
    damage_min = (
        round(game_state.p2.trained_stats["hp"] / 2)
        if criteria == "2hko"
        else game_state.p2.trained_stats["hp"]
    )

    # Store original EVs to restore later
    original_evs = game_state.p1.evs.copy() if game_state.p1.evs else None

    # Find optimal EVs through binary search
    low = 0
    high = evs_remaining
    optimal_evs = None

    while low <= high:
        mid = (low + high) // 4 * 4  # Ensure we're using multiples of 4
        game_state.p1.retrain(stat=higher_stat_name, ev=mid)
        result = game_state.execute_action()
        min_damage = (
            result[0] if isinstance(result, tuple) else result.get("min_damage", 0)
        )

        if min_damage >= damage_min:
            optimal_evs = mid
            high = mid - 4
        else:
            low = mid + 4

    # Restore original EVs
    if original_evs:
        for stat, ev in original_evs.items():
            game_state.p1.retrain(stat=stat, ev=ev)

    # If no solution found, use max EVs
    if optimal_evs is None:
        optimal_evs = evs_remaining

    results.append(
        {
            "evs_invested": optimal_evs,
            "stat": higher_stat_name,
            "training": "optimal",
            "criteria": criteria,
        }
    )

    return results
