import os
import json
import random
import numpy as np
import tensorflow as tf

from keras import Model
from keras.layers import LSTM, GRU, Dense, Embedding, Input, Concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

from prepare_data import MUSIC_EVENT_PROPERTIES

def set_random_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

def load_prepared_data(folder):
    dataset_name = ["train", "validation", "test"]
    inputs = {}
    outputs = {}

    for dataset in dataset_name:
        inputs[dataset] = {}
        outputs[dataset] = {}

        for property_name in MUSIC_EVENT_PROPERTIES:
            input_file_name = f"X_{property_name}_{dataset}.npy"
            output_file_name = f"y_{property_name}_{dataset}.npy"

            inputs[dataset][property_name] = np.load(os.path.join(folder, input_file_name))
            outputs[dataset][property_name] = np.load(os.path.join(folder, output_file_name))

    with open(os.path.join(folder, "vocabularies.json"), "r") as file:
        vocabs = json.load(file)

    return inputs, outputs, vocabs
    

def build_model(vocabularies, sequence_length, embedding_dim=64, recurrent_type="lstm", recurrent_units=128, 
                optimizer="adam", second_layer_units=None, learning_rate=0.001):
    
    network_inputs = {}
    network_outputs = {}
    embedded_inputs = []
    losses = {}
    metrics = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        vocabulary_size = len(vocabularies[property_name])
        property_input = Input(shape=(sequence_length,), name=f"{property_name}_input", dtype="int32")
        property_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, name=f"{property_name}_embedding")(property_input)
        network_inputs[f"{property_name}_input"] = property_input
        embedded_inputs.append(property_embedding)

    combined_embeddings = Concatenate(axis=-1, name="combined_embedding")(embedded_inputs)

    if recurrent_type == "lstm":
        if second_layer_units is None:
            recurrent_output = LSTM(recurrent_units, name="lstm")(combined_embeddings)
        else:
            first_lstm_output = LSTM(recurrent_units, return_sequences=True, name="first_lstm")(combined_embeddings)
            recurrent_output = LSTM(second_layer_units, name="second_lstm")(first_lstm_output)
    elif recurrent_type == "gru":
        if second_layer_units is None:
            recurrent_output = GRU(recurrent_units, name="gru")(combined_embeddings)
        else:
            first_gru_output = GRU(recurrent_units, return_sequences=True, name="first_gru")(combined_embeddings)
            recurrent_output = GRU(second_layer_units, name="second_gru")(first_gru_output)
    else:
        raise ValueError(f"Unknown recurrent type: {recurrent_type}")

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    for property_name in MUSIC_EVENT_PROPERTIES:
        vocabulary_size = len(vocabularies[property_name])
        property_output = Dense(vocabulary_size, activation="softmax", name=f"{property_name}_output")(recurrent_output)
        network_outputs[f"{property_name}_output"] = property_output
        output_name = f"{property_name}_output"
        losses[output_name] = "sparse_categorical_crossentropy"
        metrics[output_name] = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, vocabulary_size), name="top3_accuracy")]

    model = Model(inputs=network_inputs, outputs=network_outputs)
    model.compile(optimizer=opt, loss=losses, metrics=metrics)
    return model

def prepare_inputs_for_model(inputs):
    return {f"{property_name}_input": inputs[property_name] for property_name in MUSIC_EVENT_PROPERTIES}

def prepare_outputs_for_model(outputs):
    return {f"{property_name}_output": outputs[property_name] for property_name in MUSIC_EVENT_PROPERTIES}

def save_training_history(training_history, history_save_path):
    history = {}

    for metric_name, metric_values in training_history.history.items():
        history[metric_name] = [float(value) for value in metric_values]

    with open(history_save_path, "w") as f:
        json.dump(history, f)

def save_test_results(test_results, test_results_path):
    results = {metric_name: float(metric_value) for metric_name, metric_value in test_results.items()}
    with open(test_results_path, "w") as f:
        json.dump(results, f)


def train_and_evaluate(composer_folder, model_save_path, sequence_length, recurrent_units=128,
                        optimizer_name='adam', learning_rate=0.001, second_layer_units=None, random_seed=42,
                        epochs=100, batch_size=32, recurrent_type="lstm", embedding_dim=64, patience=10):
    
    set_random_seed(random_seed)

    print(f"\nUcitavanje podataka iz {composer_folder}")
    inputs, outputs, vocabs = load_prepared_data(composer_folder)

    X_train = prepare_inputs_for_model(inputs["train"])
    y_train = prepare_outputs_for_model(outputs["train"])

    X_validation = prepare_inputs_for_model(inputs["validation"])
    y_validation = prepare_outputs_for_model(outputs["validation"])

    X_test = prepare_inputs_for_model(inputs["test"])
    y_test = prepare_outputs_for_model(outputs["test"])

    print(f"Duzina sekvence: {sequence_length}")
    print(f"Broj rekurentnih jedinica: {recurrent_units}")
    print(f"Embedding dimenzija: {embedding_dim}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {learning_rate}")

    model = build_model(
        vocabularies=vocabs, 
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        recurrent_type=recurrent_type, 
        second_layer_units=second_layer_units, 
        recurrent_units=recurrent_units, 
        optimizer=optimizer_name, 
        learning_rate=learning_rate
    )
    model.summary()

    model_folder = os.path.dirname(model_save_path)
    if model_folder:
        os.makedirs(model_folder, exist_ok=True)

    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(filepath=model_save_path, monitor="val_loss", save_best_only=True, verbose=1)

    print(f"Treniranje modela: maksimalno {epochs} epoha")
    training_history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_validation, y_validation),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    print("Evaluacija najboljeg modela na test skupu")
    test_results = model.evaluate(X_test, y_test, return_dict=True, verbose=1)
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value:.4f}")

    model_file_name = os.path.splitext(os.path.basename(model_save_path))[0]
    history_save_path = os.path.join(model_folder, f"{model_file_name}_history.json")
    test_results_path = os.path.join(model_folder, f"{model_file_name}_test_results.json")

    save_training_history(training_history, history_save_path)
    save_test_results(test_results, test_results_path)

    print(f"Najbolji model sacuvan: {model_save_path}")
    print(f"Istorija sacuvana: {history_save_path}")
    print(f"Test rezultati sacuvani: {test_results_path}")

    return training_history, test_results


if __name__ == "__main__":

    sequences_length = [16, 32]
    composers = ["bach", "schubert"]

    for composer in composers:
        for sequence_length in sequences_length:

            train_and_evaluate(
                composer_folder=f"data/prepared/{composer}/seq_{sequence_length}",
                model_save_path=f"models/{composer}_seq_{sequence_length}.keras",
                sequence_length=sequence_length,
            )

    print("Treniranje svih modela je zavrseno")