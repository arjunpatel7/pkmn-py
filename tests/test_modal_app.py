# tests for the modal application proper
import pytest
import modal
import pandas as pd
from utils.calculations import read_in_pokemon, formatted_speed_check
from tqdm import tqdm
from utils.base_stat_chat import classify_intent

# we need to test the inference endpoint from Modal

# we need to test that the calc prompted from the inference endpoint ends up correct

# we need to test the intent classifier from Cohere

# We just intend to test these endpoints to make sure they are working properly


@pytest.fixture
def test_data():
    # note to self, include list of outcomes and create another fixture for the intent classifier

    # then, incorporate these test into a github action
    list_of_data = [
        "Will Finneon with 230 speed evs outspeed Meowth with 215 speed evs",
        "Does 252 Salamence outspeed 248 Talonflame?",
        "Does max speed Iron Bundle outspeed max speed Flutter Mane?",
        "Calculate if +2 124 Fuecoco outspeeds +2 4 Quaxly",
    ]

    # final speeds of each mon in the test set
    list_of_final_speeds_p1 = [115, 152, 188, 144]

    list_of_final_speeds_p2 = [137, 177, 187, 142]

    # the faster pokemon in each test case
    list_of_faster_pokemon = ["meowth", "talonflame", "iron bundle", "fuecoco"]

    df = pd.DataFrame(
        {
            "query": list_of_data,
            "faster_pokemon": list_of_faster_pokemon,
            "p1_final_speed": list_of_final_speeds_p1,
            "p2_final_speed": list_of_final_speeds_p2,
        }
    )

    return df


@pytest.fixture
def pokemon_data():
    pokemons = read_in_pokemon("./data/gen9_pokemon.jsonl")
    return pokemons


@pytest.fixture
def intent_classification_data():

    list_of_data = [
        "Will Finneon with 230 speed evs outspeed Meowth with 215 speed evs",
        "Does max speed Salamence outspeed 248 Talonflame?",
        "Does max speed Iron Bundle outspeed max speed Flutter Mane?",
        "Calculate if +2 124 Fuecoco outspeeds +2 4 Quaxly",
        "Does max speed Iron Hands outspeed +1 4 Ting-Lu",
        "What is the base special defense of Meowth?",
        "What is the base attack of Salamence?",
        "What is the base special attack of Talonflame?",
        "What is the base hp of Flutter Mane?",
        "What are the five fastest pokemon in the game?",
        "Can you tell me about the World of Pokemon?",
        "what is the best pokemon in the game?",
    ]

    list_of_classes = [
        "speed check",
        "speed check",
        "speed check",
        "speed check",
        "speed check",
        "bst check",
        "bst check",
        "bst check",
        "bst check",
        "bst check",
        "unrelated",
        "unrelated",
    ]

    df = pd.DataFrame({"query": list_of_data, "intent": list_of_classes})
    return df


def get_inference(dat):
    extract = modal.Function.lookup("pkmn-py", "run_inference")
    # call run_inference remotely on modal
    result = extract.call(dat)
    return result


def test_speed_calc(test_data, pokemon_data):
    faster_pokemon = []
    p1_final_speed = []
    p2_final_speed = []
    for d in tqdm(test_data["query"]):
        inference = get_inference(d)
        # call the speed calc function locally
        _, result, _ = formatted_speed_check(inference, pokemon_data)
        p1_final_speed.append(result["p1_final_speed"])
        p2_final_speed.append(result["p2_final_speed"])
        faster_pokemon.append(
            result["p1"]
            if result["p1_final_speed"] > result["p2_final_speed"]
            else result["p2"]
        )

    # check that each list is equal to the corresponding column in test data

    c2 = p1_final_speed == test_data["p1_final_speed"].tolist()
    c3 = p2_final_speed == test_data["p2_final_speed"].tolist()

    print(c2, c3)
    print(p1_final_speed)
    assert c2 and c3


def test_intent_classifier(intent_classification_data):
    # we'll use the intent
    predicted_intents = []
    for d in tqdm(intent_classification_data["query"]):
        intent = classify_intent(d)
        predicted_intents.append(intent)
    print(predicted_intents)

    # check that each list is equal to the corresponding column in test data
    c1 = predicted_intents == intent_classification_data["intent"].tolist()
    assert c1
