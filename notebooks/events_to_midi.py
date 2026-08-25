import os
import json
import traceback
from music21 import stream, instrument, note, chord, tempo

MINIMUM_DURATION = 0.125

def load_music_events(json_path):
    with open(json_path, "r") as f:
        music_events = json.load(f)

    return music_events

def create_instrument(instrument_name):
    try:
        return instrument.fromString(instrument_name)
    except Exception:
        unknown_instrument = instrument.Instrument()
        unknown_instrument.instrumentName = instrument_name
        return unknown_instrument

def create_parts(music_events):
    parts = {}

    for music_event in music_events:
        if music_event["type"] == "END":
            continue

        part_number = int(music_event["part"])
        if part_number in parts:
            continue

        instrument_name = str(music_event["instrument"])
        music_part = stream.Part(id=f"part_{part_number}")
        music_part.partName = instrument_name
        part_instrument = create_instrument(instrument_name)
        music_part.insert(0, part_instrument)
        parts[part_number] = music_part

    return parts

def create_music_element(music_event):
    music_event_type = music_event["type"]
    pitch = music_event["pitch"]

    if music_event_type == "NOTE":
        music_element = note.Note(pitch)
    elif music_event_type == "CHORD":
        music_element = chord.Chord(pitch.split("."))
    elif music_event_type == "REST":
        music_element = note.Rest()
    else:
        return None
    
    duration = float(music_event["duration"])
    if duration <= 0:
        duration = MINIMUM_DURATION

    music_element.duration.quarterLength = duration
    return music_element

def add_tempo_changes(score, tempo_changes):
    for tempo_start, beats_per_minute in tempo_changes:
        tempo_mark = tempo.MetronomeMark(number=beats_per_minute)
        score.insert(tempo_start, tempo_mark)

def events_to_midi(music_events, output_path):
    score = stream.Score()
    parts = create_parts(music_events)

    current_event_start = 0.0
    previous_tempo = None
    tempo_changes = []
    number_of_added_events = 0
    number_of_skipped_events = 0

    for music_event in music_events:
        if music_event["type"] == "END":
            break

        time_shift = float(music_event["time_shift"])
        current_event_start += time_shift
        current_tempo = int(float(music_event["tempo"]))

        if current_tempo != previous_tempo:
            tempo_changes.append((current_event_start, current_tempo))
            previous_tempo = current_tempo

        try:
            part_number = int(music_event["part"])
            music_element = create_music_element(music_event)

            if music_element is None:
                number_of_skipped_events += 1
                continue

            parts[part_number].insert(current_event_start, music_element)
            number_of_added_events += 1

        except Exception:
            number_of_skipped_events += 1
            print(f"Preskocen je muzicki dogadjaj: {music_event}")
            traceback.print_exc()
            break
    
    for part_number in sorted(parts):
        score.insert(0, parts[part_number])

    add_tempo_changes(score, tempo_changes)
    output_folder = os.path.dirname(output_path)
    if output_path:
        os.makedirs(output_folder, exist_ok=True)

    score.write("midi", fp=output_path)

    print()
    print(f"MIDI fajl sacuvan: {output_path}")
    print(f"Dodato muzickih dogadjaja: {number_of_added_events}")
    print(f"Preskoceno muzickih dogadjaja: {number_of_skipped_events}")
    print(f"Broj delova: {len(parts)}")

def convert_json_to_midi(json_path, output_path):
    print(f"\nUcitavanje json fajla: {json_path}")
    music_events = load_music_events(json_path)
    print(f"\nPronadjeno {len(music_events)} muzickih dogadjaja")
    events_to_midi(music_events, output_path)

if __name__ == "__main__":
    convert_json_to_midi(
        json_path="data/parsed/bach/completed/bwv1.6.json", 
        output_path="generated/reconstructed_bwv1.6.mid"
    )