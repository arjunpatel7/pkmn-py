# tests for the modal application proper
import pytest
import modal
import pandas as pd
from app.utils.calculations import read_in_pokemon, formatted_speed_check
from tqdm import tqdm
from app.utils.base_stat_chat import classify_intent
import streamlit as st
import cohere
from cohere.error import CohereAPIError
import time

from supabase import create_client

sb_url = st.secrets["SUPABASE_URL"]
sb_key = st.secrets["SUPABASE_KEY"]


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
def dmg_data():
    # read in the first 5 rows of the eval damage calc data jsonl
    data = [
        "train Naclstack using scratch to 2hko Cinderace.",
        "train Basculin Red Striped using pay-day to ohko Glimmet.",
        "What are the optimal evs for Eelektross to 1hko Cacturne using fire-punch?",
        "train Barraskewda using pay-day to ohko Garchomp.",
    ]
    return data


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
        "What is the base special defense of Flutter Mane?",
        "Can you tell me about the World of Pokemon?",
    ]

    list_of_classes = [
        "speed check",
        "speed check",
        "speed check",
        "speed check",
        "speed check",
        "sql query",
        "sql query",
        "sql query",
        "sql query",
        "unrelated",
    ]

    df = pd.DataFrame({"query": list_of_data, "intent": list_of_classes})
    return df


def get_inference(dat):
    extract = modal.Function.lookup("pkmn-py", "run_inference")
    # call run_inference remotely on modal
    result = extract.remote(dat)
    return result


def get_batch_inference(dat):
    extract = modal.Function.lookup("pkmn-py", "run_batch_inference")
    result = extract.remote(dat)
    return result


def get_dmg_inference_batch(dat):
    extract = modal.Function.lookup("pkmn-dmg", "run_batch_inference")
    result = extract.remote(dat)
    return result


def test_speed_calc(test_data, pokemon_data):
    # time the inference

    start_time = time.time()
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
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    # time per input
    print(f"Time per input: {(end_time - start_time) / len(test_data)}")


def test_batch_inference(test_data, pokemon_data):
    # similar to test_speed_calc, but we'll use the batch inference endpoint

    # time the inference
    start_time = time.time()

    faster_pokemon = []
    p1_final_speed = []
    p2_final_speed = []
    processed_mons = get_batch_inference(test_data["query"].tolist())
    print(processed_mons)
    for d in tqdm(processed_mons):
        # call the speed calc function locally
        _, result, _ = formatted_speed_check(d, pokemon_data)
        p1_final_speed.append(result["p1_final_speed"])
        p2_final_speed.append(result["p2_final_speed"])
        faster_pokemon.append(
            result["p1"]
            if result["p1_final_speed"] > result["p2_final_speed"]
            else result["p2"]
        )
    c2 = p1_final_speed == test_data["p1_final_speed"].tolist()
    c3 = p2_final_speed == test_data["p2_final_speed"].tolist()

    print(c2, c3)
    print(p1_final_speed)
    assert c2 and c3
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    # time per input
    print(f"Time per input: {(end_time - start_time) / len(test_data)}")


def test_dmg_batch_inference(dmg_data):
    # we'll do this, but we'll use the batch inference endpoint for the damage calc
    # but we won't check the results, we'll just check that the endpoint is working
    # and that the time per input is reasonable

    # time the inference
    start_time = time.time()
    processed_mons = get_dmg_inference_batch(dmg_data)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    # time per input
    print(f"Time per input: {(end_time - start_time) / len(dmg_data)}")
    assert processed_mons is not None


def test_cohere_endpoint():
    # just checks that the cohere endpoint is working
    prompt = "Testing one two three"
    co = cohere.Client(st.secrets["COHERE_API_KEY"])  # This is your trial API key
    try:
        response = co.classify(
            model="bfb1f19a-afaa-4faf-89db-f35df53f9de6-ft", inputs=[prompt]
        )
        assert response is not None
    except CohereAPIError:
        pytest.fail("Cohere API is down")


def test_intent_classifier(intent_classification_data):
    # we'll use the intent
    predicted_intents = []
    for d in tqdm(intent_classification_data["query"]):
        intent = classify_intent(d)
        predicted_intents.append(intent)
    print(predicted_intents)

    # check that each list is equal to the corresponding column in test data
    c1 = sum(predicted_intents == intent_classification_data["intent"])
    assert c1 / len(predicted_intents) >= 0.8


def test_supabase_active():
    supabase_client = create_client(sb_url, sb_key)

    try:
        # Make a simple query to validate connectivity
        data = (
            supabase_client.table("speed-checks-logs").select("id").limit(1).execute()
        )
        assert data is not None
    except Exception as e:
        pytest.fail(f"Got exception connecting to Supabase: {e}")


# Write a test that attempts to check that all pokemon names
# in our database are parsed correctly by our LLM
# Do this in batch so that inference time is faster


def test_stress_pokemon_name_extraction(pokemon_data):
    # for every pokemon in pokemon_data
    # generate 10 speed calc queries
    # combine all the queries
    # run batch inference job on the queries

    # check if the queries are properly parsed

    # report statistics on how many queries were parsed correctly per mon
    # fail if average query passing rate is below 80%
    # write out the names of pokemon that failed 90% or more of the time

    return


# idea for a test

# generate a set of random calcs, and check how many are successful and unsuccessful

# somehow determine what kinds of errors are being made on the unsucccessful ones
# maybe by checking parameters like the evs, pokemon names, etc

# return a report of the errors and the success rate
