import os
import json
import numpy as np
import tensorflow as tf

from prepare_data import MUSIC_EVENT_PROPERTIES
from events_to_midi import events_to_midi

END_VALUE = "END"
UNKNOWN_VALUE = "UNK"

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_json(data, path):
    output_folder = os.path.dirname(path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def create_inverse_vocabularies(vocabs):
    inverse_vocabs = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        inverse_vocabs[property_name] = {int(number): value for value, number in vocabs[property_name].items()}
    return inverse_vocabs

def get_encoded_value(property_value, property_vocab):
    property_value = str(property_value)

    if property_value in property_vocab:
        return property_vocab[property_value]
    return property_vocab[UNKNOWN_VALUE]

def create_part_instrument_mapping(music_events):
    part_instruments = {}

    for music_event in music_events:
        if music_event["type"] == END_VALUE:
            continue

        part_number = int(music_event["part"])
        instrument_name = str(music_event["instrument"])
        part_instruments[part_number] = instrument_name

    return part_instruments

def prepare_sequence_for_model(music_events, vocabularies, sequence_length):
    selected_events = music_events[-sequence_length:]
    model_inputs = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        property_vocabulary = vocabularies[property_name]
        encoded_values = []

        for music_event in selected_events:
            property_value = music_event[property_name]
            encoded_values.append(get_encoded_value(property_value, property_vocabulary))

        model_inputs[f"{property_name}_input"] = np.array([encoded_values], dtype=np.int32)
    return model_inputs

def convert_predictions_to_dictionary(model, predictions):
    if isinstance(predictions, dict):
        return predictions
    return {output_name: prediction for output_name, prediction in zip(model.output_names, predictions)}

def convert_number(value):
    try:
        number = float(value)

        if number.is_integer():
            return int(number)
        return number
    except Exception:
        return value

def choose_value(probabilities, inverse_vocabularies, temperature=0.8, forbidden_values=None, required_condition=None):
    probabilities = np.array(probabilities, dtype=np.float64)

    if forbidden_values is None:
        forbidden_values = []

    allowed_numbers = []
    for number, value in inverse_vocabularies.items():
        if value in forbidden_values:
            continue

        if required_condition is not None:
            if not required_condition(value):
                continue
        
        allowed_numbers.append(number)

    if len(allowed_numbers) == 0:
        raise ValueError("Nema dozvoljenih vrednosti za predvidjanje")
    
    allowed_probabilities = probabilities[allowed_numbers]
    allowed_probabilities = np.clip(allowed_probabilities, 1e-10, 1.0)

    logs = np.log(allowed_probabilities) / temperature
    logs -= np.max(logs)
    allowed_probabilities = np.exp(logs)
    allowed_probabilities /= np.sum(allowed_probabilities)

    select_number = np.random.choice(allowed_numbers, p=allowed_probabilities)
    return inverse_vocabularies[select_number]

def choose_music_event_type(predictions, inverse_vocabularies, temperature):
    type_probabilities = predictions["type_output"][0]
    allowed_types = {"NOTE", "CHORD", "REST", END_VALUE}
    return choose_value(
            probabilities=type_probabilities, 
            inverse_vocabularies=inverse_vocabularies["type"], 
            temperature=temperature, 
            forbidden_values=[UNKNOWN_VALUE], 
            required_condition=lambda value:value in allowed_types
        )

def choose_pitch(music_event_type, predictions, inverse_vocabularies, temperature):
    if music_event_type == "REST":
        return "NONE"

    pitch_probabilities = predictions["pitch_output"][0]
    if music_event_type == "NOTE":
        pitch_condition = lambda value: value != "NONE" and "." not in value
    else:
        pitch_condition = lambda value: "." in value

    return choose_value(
            probabilities=pitch_probabilities, 
            inverse_vocabularies=inverse_vocabularies["pitch"], 
            temperature=temperature, 
            forbidden_values=[UNKNOWN_VALUE, END_VALUE], 
            required_condition=pitch_condition
        )

def choose_property(property_name, predictions, inverse_vocabularies, temperature):
    probabilities = predictions[f"{property_name}_output"][0]
    return choose_value(
            probabilities=probabilities, 
            inverse_vocabularies=inverse_vocabularies[property_name], 
            temperature=temperature, 
            forbidden_values=[UNKNOWN_VALUE, END_VALUE]
        )

def calculate_measure_number(previous_event, position_in_measure):
    previous_measure = int(previous_event.get("measure", 1))
    previous_position = float(previous_event.get("position_in_measure", 0.0))

    if position_in_measure < previous_position:
        return previous_measure + 1
    
    return previous_measure

def create_next_music_event(model,  current_events, vocabularies, inverse_vocabularies, 
                            part_instruments, sequence_length, temperature):
    model_inputs = prepare_sequence_for_model(current_events, vocabularies, sequence_length)
    predictions = model.predict(model_inputs, verbose=0)
    predictions = convert_predictions_to_dictionary(model, predictions)

    music_event_type = choose_music_event_type(predictions, inverse_vocabularies, temperature)
    if music_event_type == END_VALUE:
        return {property_name: END_VALUE for property_name in MUSIC_EVENT_PROPERTIES}
    
    pitch = choose_pitch(music_event_type, predictions, inverse_vocabularies, temperature)
    duration = convert_number(choose_property("duration", predictions, inverse_vocabularies, temperature))
    time_shift = convert_number(choose_property("time_shift", predictions, inverse_vocabularies, temperature))
    tempo_value = convert_number(choose_property("tempo", predictions, inverse_vocabularies, temperature))
    part_number = convert_number(choose_property("part", predictions, inverse_vocabularies, temperature))
    position_in_measure = convert_number(choose_property("position_in_measure", predictions, inverse_vocabularies, temperature))

    part_number = int(part_number)
    if part_number in part_instruments:
        instrument_name = part_instruments[part_number]
    else:
        instrument_name = choose_property("instrument", predictions, inverse_vocabularies, temperature)
        part_instruments[part_number] = instrument_name

    previous_event = current_events[-1]
    measure_number = calculate_measure_number(previous_event, float(position_in_measure))
    
    return {
        "type": music_event_type,
        "pitch": pitch,
        "duration": duration,
        "time_shift": time_shift,
        "tempo": tempo_value,
        "instrument": instrument_name,
        "part": part_number,
        "measure": measure_number,
        "position_in_measure": position_in_measure
    }

def generate_music(model_path, vocabularies_path, unfinished_composition_path, 
                   generated_json_path, generated_midi_path, sequence_length=16,
                   max_generated_events=2000, temperature=0.8, random_seed=42):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    print(f"Ucitavanje modela {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"Ucitavanje recnika {vocabularies_path}")
    vocabs = load_json(vocabularies_path)
    inverse_vocabs = create_inverse_vocabularies(vocabs)

    print(f"Ucitavanje nedovrsenih kompozicija: {unfinished_composition_path}")
    unfinished_events = load_json(unfinished_composition_path)
    unfinished_events = [music_event for music_event in unfinished_events if music_event["type"] != END_VALUE]
    if len(unfinished_events) < sequence_length:
        raise ValueError("Nedovrsena kompozicija nema dovoljno dogadjaja za pocetnu sekvencu")
    
    part_instruments = create_part_instrument_mapping(unfinished_events)
    all_events = list(unfinished_events)
    generated_events = []

    print("\nPocetak generisanja")
    print(f"Maksimalan broj novih dogadjaja je {max_generated_events}")
    print(f"Temperatura {temperature}")

    for event_number in range(max_generated_events):
        next_music_event = create_next_music_event(
            model=model, 
            current_events=all_events, 
            vocabularies=vocabs, 
            inverse_vocabularies=inverse_vocabs, 
            part_instruments=part_instruments, 
            sequence_length=sequence_length, 
            temperature=temperature
        )

        if next_music_event["type"] == END_VALUE:
            print(f"Model je predvideo END nakon {event_number} novih dogadjaja")
            break

        generated_events.append(next_music_event)
        all_events.append(next_music_event)

        if(event_number + 1) % 50 == 0:
            print(f"Generisano {event_number + 1} dogadjaja")

    save_json(generated_events, generated_json_path)
    events_to_midi(all_events, generated_midi_path)

    print(f"\nGenerisani JSON sacuvan: {generated_json_path}")
    print(f"\nGenerisani MIDI sacuvan: {generated_midi_path}")
    print(f"\nUkupno generisno {len(generated_events)} muzickih dogadjaja")


if __name__ == "__main__":

    unfinished_compositions = [
        {
            "composer": "bach",
            "composition_name": "unfinished_fugue",
            "json_path": "data/parsed/bach/unfinished/unfinished_fugue.json",
            "max_generated_events": 2000
        },
        {
            "composer": "schubert",
            "composition_name": "d759_movement3_sketch",
            "json_path": "data/parsed/schubert/unfinished/d759_movement3_sketch.json",
            "max_generated_events": 20000
        }
    ]

    sequence_lengths = [16, 32]

    for composition in unfinished_compositions:
        composer = composition["composer"]
        composition_name = composition["composition_name"]
        max_generated_events = composition["max_generated_events"]

        for sequence_length in sequence_lengths:
            print("=" * 60)
            print(f"Generisanje kompozicije: {composition_name}")
            print(f"Kompozitor: {composer.capitalize()}")
            print(f"Duzina sekvence: {sequence_length}")
            print("=" * 60)

            generate_music(
                model_path=f"models/{composer}_seq_{sequence_length}.keras",
                vocabularies_path=f"data/prepared/{composer}/seq_{sequence_length}/vocabularies.json",
                unfinished_composition_path=composition["json_path"],
                generated_json_path=f"generated/{composer}/{composition_name}_seq_{sequence_length}_generated.json",
                generated_midi_path=f"generated/{composer}/{composition_name}_seq_{sequence_length}_generated.mid",
                sequence_length=sequence_length,
                max_generated_events=max_generated_events
            )

    print("\nGenerisanje svih kompozicija je zavrseno")