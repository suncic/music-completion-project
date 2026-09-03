import os
import math
import numpy as np

from collections import Counter
from prepare_data import MUSIC_EVENT_PROPERTIES
from generate_music import load_json, save_json

def save_text_report(text_report, path):
    output_folder = os.path.dirname(path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    with open(path, "w") as f:
        f.write(text_report)

def load_compositions_from_folder(folder_path):
    music_events = []

    if not os.path.exists(folder_path):
        print(f"Folder sa izgenerisanim kompozicijama ne postoji: {folder_path}")
        return music_events
    
    files = sorted(file_name for file_name in os.listdir(folder_path) if file_name.endswith(".json"))
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        compositions_events = load_json(file_path)

        for music_event in compositions_events:
            if music_event["type"] != "END":
                music_events.append(music_event)

    return music_events

def create_event_signature(music_event):
    return {
        str(music_event.get("type")),
        str(music_event.get("pitch")),
        str(music_event.get("duration")),
        str(music_event.get("time_shift")),
        str(music_event.get("part"))
    }

def calculate_repetition_ratio(music_events):
    if len(music_events) < 2:
        return 0.0
    
    repeated_events = 0
    for event_number in range(1, len(music_events)):
        previous_signature = create_event_signature(music_events[event_number-1])
        current_signature = create_event_signature(music_events[event_number])
        if current_signature == previous_signature:
            repeated_events += 1

    return repeated_events / (len(music_events)-1)

def calculate_basic_statistics(music_events):
    if len(music_events) == 0:
        return {
            "number_of_events": 0,
            "unique_pitches": 0,
            "number_of_parts": 0,
            "number_of_instruments": 0,
            "average_duration": 0.0,
            "average_time_shift": 0.0,
            "zero_time_shift_ratio": 0.0,
            "repetition_ratio": 0.0
        }
    
    pitches = {str(music_event["pitch"]) for music_event in music_events if music_event["pitch"] not in ["NONE", "UNK", "END"]}
    parts = {str(music_event["part"]) for music_event in music_events}
    instruments = {str(music_event["instrument"]) for music_event in music_events}
    durations = [float(music_event["duration"]) for music_event in music_events]
    time_shifts = [float(music_event["time_shift"]) for music_event in music_events]
    zero_time_shift = sum(1 for time_shift in time_shifts if time_shift == 0)

    return {
        "number_of_events": len(music_events),
        "unique_pitches": len(pitches),
        "number_of_parts": len(parts),
        "number_of_instruments": len(instruments),
        "average_duration": float(np.mean(durations)),
        "average_time_shift": float(np.mean(time_shifts)),
        "zero_time_shift_ratio": zero_time_shift / len(time_shifts),
        "repetition_ratio": calculate_repetition_ratio(music_events)
    }

def find_invalid_events(music_events):
    invalid_events = []

    for event_number, music_event in enumerate(music_events):
        problems = []

        music_event_type = music_event.get("type")
        pitch = str(music_event.get("pitch"))
        try:
            duration = float(music_event.get("duration"))
            if duration <= 0:
                problems.append("Trajanje nije pozitivno")
        except Exception:
            problems.append("Trajanje nije broj")

        try:
            time_shift = float(music_event.get("time_shift"))
            if time_shift < 0:
                problems.append("Time shift nije pozitivno")
        except Exception:
            problems.append("Time shift nije broj")

        if music_event_type == "NOTE":
            if pitch in ["NOTE", "END", "UNK"]:
                problems.append("NOTE nema ispravnu visinu tona")

            if "." in pitch:
                problems.append("NOTE sadrzi vise visina tonova (ponasa se kao akord)")
        elif music_event_type == "CHORD":
            if "." not in pitch:
                problems.append("CHORD nema vise visina tonova")
        elif music_event_type == "REST":
            if pitch != "NONE":
                problems.append("REST ima visinu tona")
        else:
            problems.append("Nepoznati tip muzickog dogadjaja")

        if music_event.get("part") is None:
            problems.append("Nedostaje broj deonice")

        if music_event.get("instrument") is None:
            problems.append("Nedostaje instrument")

        if len(problems) > 0:
            invalid_events.append(
            {
                "event_number": event_number,
                "music_event": music_event,
                "problems": problems
            }
        )

    return invalid_events

def create_property_distribution(music_events, property_name):
    property_values = [str(music_event[property_name]) for music_event in music_events if property_name in music_event]
    value_counts = Counter(property_values)
    total_values = len(property_values)

    if total_values == 0:
        return {}
    
    return {value: count / total_values for value, count in value_counts.items()}

def calculate_jensen_shannon_distance(first_distribution, second_distribution):
    all_values = sorted(set(first_distribution.keys()) | set(second_distribution.keys()))
    if len(all_values) == 0:
        return 0.0
    
    first_probabilities = np.array([first_distribution.get(value, 0.0) for value in all_values], dtype=np.float64)
    second_probabilities = np.array([second_distribution.get(value, 0.0) for value in all_values], dtype=np.float64)
    first_probabilities /= np.sum(first_probabilities)
    second_probabilities /= np.sum(second_probabilities)
    middle_probabilities = (first_probabilities + second_probabilities) / 2

    first_divergence = 0.0
    second_divergence = 0.0
    for first, second, middle in zip(first_probabilities, second_probabilities, middle_probabilities):
        if first > 0:
            first_divergence += first * math.log2(first / middle)

        if second > 0:
            second_divergence += second * math.log2(second / middle)

    divergence = (first_divergence + second_divergence) / 2
    return float(math.sqrt(divergence))

def calculate_distribution_distances(generated_events, completed_compositions_events):
    distances = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        generated_distribution = create_property_distribution(generated_events, property_name)
        reference_distribution = create_property_distribution(completed_compositions_events, property_name)
        distances[property_name] = calculate_jensen_shannon_distance(generated_distribution, reference_distribution)
    return distances

def create_conclusions(statistics, invalid_events, distribution_distances):
    conclusions = []

    number_of_events = statistics["number_of_events"]
    if number_of_events == 0:
        conclusions.append("Model nije generisao nijedan muzicki dogadjaj")
        return conclusions

    invalid_ratio = len(invalid_events) / number_of_events
    if invalid_ratio == 0:
        conclusions.append("Svi generisani dogadjaji su tehnicki ispravni")
    elif invalid_ratio <= 0.05:
        conclusions.append("Vecina generisanih dogadjaja je tehnicki ispravna, ali postoji mali broj gresaka")
    else:
        conclusions.append("Generisani nastavak sadrzi znacajan broj tehnicki neispravnih dogadjaja")

    repetition_ratio = statistics["repetition_ratio"]
    if repetition_ratio <= 0.10:
        conclusions.append("Nije primeceno prekomerno neposredno ponavljanje istih dogadjaja")
    elif repetition_ratio <= 0.30:
        conclusions.append("Postoji umereno ponavljanje istih muzickih dogadjaja")
    else:
        conclusions.append("Model cesto ponavlja iste muzicke dogadjaje")

    average_distance = float(np.mean(list(distribution_distances.values())))
    if average_distance <= 0.20:
        conclusions.append("Raspodele muzickih dogadjaja su veoma slicne zavrsenim delima kompozitora")
    elif average_distance <= 0.40:
        conclusions.append("Raspodele muzickih dogadjaja pokazuju umerenu slicnost sa zavrsenim delima kompozitora")
    else:
        conclusions.append("Raspodele muzickih dogadjaja se znatno razlikuju od zavrsenih dela kompozitora")

    
    if statistics["unique_pitches"] < 5:
        conclusions.append("Generisani nastavak koristi veoma mali broj razlicitih visina tonova")
    else:
        conclusions.append("Generisani nastavak koristi vise razlicitih visina tonova")

    return conclusions

def calculate_model_score(statistics, invalid_events, distribution_distances):
    number_of_events = max(statistics["number_of_events"], 1)
    invalid_ratio = len(invalid_events) / number_of_events
    average_distance = float(np.mean(list(distribution_distances.values())))
    repetition_ratio = statistics["repetition_ratio"]
    return float(average_distance + repetition_ratio + 2 * invalid_ratio)

def evaluate_generated_composition(composer, composition_name, sequence_length, generated_json_path, completed_compositions_folder):
    print(f"Evaluacija {composition_name} sekvenca: {sequence_length}")
    generated_events = load_json(generated_json_path)
    completed_compositions_events = load_compositions_from_folder(completed_compositions_folder)
    statistics = calculate_basic_statistics(generated_events)
    invalid_events = find_invalid_events(generated_events)
    distribution_distances = calculate_distribution_distances(generated_events, completed_compositions_events)
    conclusions = create_conclusions(statistics, invalid_events, distribution_distances)
    model_score = calculate_model_score(statistics, invalid_events, distribution_distances)
    return {
        "composer": composer,
        "composition_name": composition_name,
        "sequence_length": sequence_length,
        "generated_json_path": generated_json_path,
        "statistics": statistics,
        "number_of_invalid_events": len(invalid_events),
        "invalid_events": invalid_events[:20],
        "distribution_distances": distribution_distances,
        "average_distribution_distance": float(np.mean(list(distribution_distances.values()))),
        "model_score": model_score,
        "conclusions": conclusions
    }

def create_text_report(results):
    report_lines = []

    report_lines.append("EVALUACIJA GENERISANIH KOMPOZICIJA")
    sorted_results = sorted(results, key=lambda result: result["model_score"])
    for result in sorted_results:
        statistics = result["statistics"]

        report_lines.append("")
        report_lines.append(f"Kompozitor: {result['composer'].capitalize()}")
        report_lines.append(f"Kompozicija: {result['composition_name']}")
        report_lines.append(f"Duzina sekvence: {result['sequence_length']}")
        report_lines.append(f"Broj generisanih dogadjaja: {statistics['number_of_events']}")
        report_lines.append(f"Broj razlicitih visina nota: {statistics['unique_pitches']}")
        report_lines.append(f"Broj deonica: {statistics['number_of_parts']}")
        report_lines.append(f"Prosecno trajanje: {statistics['average_duration']:.4f}")
        report_lines.append(f"Prosecan time shift: {statistics['average_time_shift']:.4f}")
        report_lines.append(f"Udeo ponavljanja: {statistics['repetition_ratio']:.4f}")
        report_lines.append(f"Broj neispravnih dogadjaja: {result['number_of_invalid_events']}")
        report_lines.append(f"Prosecna udaljenost raspodela: {result['average_distribution_distance']:.4f}")
        report_lines.append(f"Zbirna ocena modela: {result['model_score']:.4f}")
        report_lines.append("Zakljucci:")
        for conclusion in result["conclusions"]:
            report_lines.append(f"- {conclusion}")

    if len(sorted_results) > 0:
        best_result = sorted_results[0]
        report_lines.append(f"Najbolji rezultat prema automatskim metrikama: {best_result['composer'].capitalize()} - {best_result['composition_name']} - sekvenca {best_result['sequence_length']}")

    return "\n".join(report_lines)

if __name__ == "__main__":

    unfinished_compositions = [
        {
            "composer": "bach",
            "composition_name": "unfinished_fugue",
            "completed_folder": "data/parsed/bach/completed"
        },
        {
            "composer": "schubert",
            "composition_name": "d759_movement3_sketch",
            "completed_folder": "data/parsed/schubert/completed"
        }
    ]

    sequence_lengths = [16, 32]

    for composition in unfinished_compositions:
        composer = composition["composer"]
        composition_name = composition["composition_name"]
        composition_results = []

        for sequence_length in sequence_lengths:
            generated_json_path = f"generated/{composer}/{composition_name}_seq_{sequence_length}_generated.json"
            if not os.path.exists(generated_json_path):
                print(f"Preskocen fajl koji ne postoji: {generated_json_path}")
                continue

            results = evaluate_generated_composition(
                composer=composer,
                composition_name=composition_name,
                sequence_length=sequence_length,
                generated_json_path=generated_json_path,
                completed_compositions_folder=composition["completed_folder"]
            )

            composition_results.append(results)

        save_json(composition_results, f"generated/{composer}/{composition_name}_evaluation_results.json")
        text_report = create_text_report(composition_results)
        save_text_report(text_report, f"generated/{composer}/{composition_name}_evaluation_report.txt")

    print("\nPravljenje izvestaja je gotovo")