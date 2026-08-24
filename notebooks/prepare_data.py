import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

MUSIC_EVENT_PROPERTIES = ["type", "pitch", "duration", "time_shift", "tempo", 
                          "instrument", "part", "position_in_measure"]
UNKNOWN_VALUE = "UNK"
END_VALUE = "END"

def create_end_event():
    return {property_name: END_VALUE for property_name in MUSIC_EVENT_PROPERTIES}

def load_compositions(parsed_folder):
    compositions = []

    if not os.path.exists(parsed_folder):
        print(f"Folder sa parsiranim kompozicijama ne postoji: {parsed_folder}")
        return compositions

    files = sorted(file for file in os.listdir(parsed_folder) if file.endswith('.json'))

    for file_name in files:
        path = os.path.join(parsed_folder, file_name)
        with open(path, 'r') as f:
            music_events = json.load(f)
        
        if len(music_events) == 0:
            continue
        
        music_events.append(create_end_event())
        compositions.append({"file_name": file_name, "music_events": music_events})

    return compositions

def split_compositions(compositions, test_size=0.1, validation_size=0.1, random_state=42):
    if len(compositions) < 3:
        raise ValueError("Potrebne su najmanje 3 kompozicije da bi bio napravljen train, validation i test skup.")

    train_validation_compositions, test_compositions = train_test_split(compositions, test_size=test_size, random_state=random_state)
    validation_ratio = validation_size / (1 - test_size)
    train_compositions, validation_compositions = train_test_split(train_validation_compositions, test_size=validation_ratio, random_state=random_state)
    return train_compositions, validation_compositions, test_compositions

def create_vocabularies(compositions):
    unique_events_values = {property_name: {UNKNOWN_VALUE} for property_name in MUSIC_EVENT_PROPERTIES}

    for composition in compositions:
        for music_event in composition["music_events"]:
            for property_name in MUSIC_EVENT_PROPERTIES:
                unique_events_values[property_name].add(str(music_event[property_name]))

    vocabularies = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        sorted_values = sorted(unique_events_values[property_name])
        vocabularies[property_name] = {value: number for number, value in enumerate(sorted_values)}

    return vocabularies

def get_value_number(property_value, property_vocabulary):
    property_value = str(property_value)

    if property_value in property_vocabulary:
        return property_vocabulary[property_value]

    return property_vocabulary[UNKNOWN_VALUE]

def prepare_sequences(compositions, vocabularies, sequence_length=16):
    network_input = {property_name: [] for property_name in MUSIC_EVENT_PROPERTIES}
    network_output = {property_name: [] for property_name in MUSIC_EVENT_PROPERTIES}

    for composition in compositions:
        music_events = composition["music_events"]

        if len(music_events) <= sequence_length:
            continue

        number_of_sequences = len(music_events) - sequence_length
        for position in range(number_of_sequences):
            input_events = music_events[position: position + sequence_length]
            next_event = music_events[position + sequence_length]

            for property_name in MUSIC_EVENT_PROPERTIES:
                property_vocabulary = vocabularies[property_name]
                encoded_input_sequence = []

                for music_event in input_events:
                    encoded_value = get_value_number(property_value=music_event[property_name], property_vocabulary=property_vocabulary)
                    encoded_input_sequence.append(encoded_value)

                output_value = get_value_number(next_event[property_name], property_vocabulary)

                network_input[property_name].append(encoded_input_sequence)
                network_output[property_name].append(output_value)

    for property_name in MUSIC_EVENT_PROPERTIES:
        network_input[property_name] = np.array(network_input[property_name], dtype=np.int32)
        network_output[property_name] = np.array(network_output[property_name], dtype=np.int32)
        print(property_name, network_input[property_name].shape, network_output[property_name].shape)

    return network_input, network_output

def save_prepared_data(output_folder, dataset_name, network_inputs, network_outputs):
    os.makedirs(output_folder, exist_ok=True)
    for property_name in MUSIC_EVENT_PROPERTIES:
        input_file_name = f"X_{property_name}_{dataset_name}.npy"
        output_file_name = f"y_{property_name}_{dataset_name}.npy"

        np.save(os.path.join(output_folder, input_file_name), network_inputs[property_name])
        np.save(os.path.join(output_folder, output_file_name), network_outputs[property_name])
  
def save_composition_split(output_folder, train_compositions, validation_compositions, test_compositions):
    composition_split = {
        "train": [composition["file_name"] for composition in train_compositions],
        "validation": [composition["file_name"] for composition in validation_compositions],
        "test": [composition["file_name"] for composition in test_compositions]
    }

    with open(os.path.join(output_folder,"composition_split.json"), "w") as f:
        json.dump(composition_split, f)


def prepare_composer(parsed_folder, output_folder, composer_name, sequence_length=16):
    print(f"Pripremanje podataka za kompozitora: {composer_name}")

    compositions = load_compositions(parsed_folder)
    vocabs = create_vocabularies(compositions)
    train_compositions, validation_compositions, test_compositions = split_compositions(compositions)

    X_train, y_train = prepare_sequences(train_compositions, vocabs, sequence_length)
    X_validation, y_validation = prepare_sequences(validation_compositions, vocabs, sequence_length)
    X_test, y_test = prepare_sequences(test_compositions, vocabs, sequence_length)

    os.makedirs(output_folder, exist_ok=True)
    save_prepared_data(output_folder, "train", X_train, y_train)
    save_prepared_data(output_folder, "validation", X_validation, y_validation)
    save_prepared_data(output_folder, "test", X_test, y_test)

    with open(os.path.join(output_folder, 'vocabularies.json'), 'w') as f:
        json.dump(vocabs, f)

    save_composition_split(output_folder, train_compositions, validation_compositions, test_compositions)

    print(f"Kompozicije za treniranje: {len(train_compositions)}")
    print(f"Kompozicije za validaciju: {len(validation_compositions)}")
    print(f"Kompozicije za testiranje: {len(test_compositions)}")
    print(f"Sekvence za treniranje: {len(X_train)}")
    print(f"Sekvence za validaciju: {len(X_validation)}")
    print(f"Sekvence za testiranje: {len(X_test)}")

if __name__ == "__main__":
    sequences_length = [16, 32]
    composers = ["bach", "schubert"]

    for composer in composers:
        print(f"===== {composer.capitalize()} =====")

        for sequence_length in sequences_length:
            print(f"Duzina sekvence: {sequence_length}")
            prepare_composer(
                parsed_folder=f"data/parsed/{composer}/completed",
                output_folder=f"data/prepared/{composer}/seq_{sequence_length}",
                composer_name=composer.capitalize(),
                sequence_length=sequence_length
            )

    print()
    print("Pripremljeni podaci za pokretanje modela")