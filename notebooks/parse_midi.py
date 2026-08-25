import os
import json
from music21 import converter, note, chord, tempo, stream

DURATIONS = [0.0, 0.125, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
TIME_SHIFTS = [0.0, 0.125, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
TEMPO_STEP = 5
DEFAULT_TEMPO = 120

def find_nearest_allowed_value(value, allowed_values):
    return min(allowed_values, key=lambda allowed_value: abs(allowed_value - value))

def round_tempo(bpm):
    return max(TEMPO_STEP, int(round(bpm / TEMPO_STEP) * TEMPO_STEP))

def get_tempo_changes(midi):
    tempo_changes = []
    flat_midi = midi.flatten()
    tempo_marks = flat_midi.getElementsByClass(tempo.MetronomeMark)

    for tempo_mark in tempo_marks:
        bpm = tempo_mark.getQuarterBPM()
        if bpm is None:
            continue
        
        tempo_changes.append((float(tempo_mark.offset), round_tempo(float(bpm))))

    if not tempo_changes or tempo_changes[0][0] > 0:
        tempo_changes.insert(0, (0.0, DEFAULT_TEMPO))

    return sorted(tempo_changes, key=lambda pair: pair[0])

def find_tempo_at_music_event_start(music_event_start, tempo_changes):
    tempo_at_current_event_start = DEFAULT_TEMPO

    for number_of_measure, new_tempo in tempo_changes:
        if number_of_measure <= music_event_start:
            tempo_at_current_event_start = new_tempo

    return tempo_at_current_event_start

def get_instrument_name(part, part_number):
    instrument = part.getInstrument(returnDefault=True)

    if part.partName:
        return str(part.partName)
    
    if instrument.instrumentName:
        return str(instrument.instrumentName)
    
    return f"Part_{part_number}"

def get_measure_information(music_event, part):
    measure = music_event.getContextByClass(stream.Measure)

    if measure is None:
        return 0, 0.0
    
    measure_number = measure.number
    measure_of_event_start = float(measure.getOffsetInHierarchy(part))
    music_event_start = float(music_event.getOffsetInHierarchy(part))
    position_in_measure = find_nearest_allowed_value(max(0.0, music_event_start - measure_of_event_start), TIME_SHIFTS)
    return measure_number, position_in_measure


def parse_midi_file(file_path):
    music_events = []

    try:
        midi = converter.parse(file_path)
        tempo_changes = get_tempo_changes(midi)

        for part_number, part in enumerate(midi.parts, start=1):
            instrument_name = get_instrument_name(part, part_number)

            for element in part.recurse().notesAndRests:
                music_event_start = float(element.getOffsetInHierarchy(part))
                duration = find_nearest_allowed_value(float(element.duration.quarterLength), DURATIONS)
                measure_number, position_in_measure = get_measure_information(element, part)

                if isinstance(element, note.Note):
                    music_event_type = "NOTE"
                    pitch = str(element.pitch)
                elif isinstance(element, chord.Chord):
                    music_event_type = "CHORD"
                    pitch = ".".join(str(value) for value in element.pitches)
                elif isinstance(element, note.Rest):
                    music_event_type = "REST"
                    pitch = "NONE"
                else:
                    continue

                music_events.append(
                    {
                        "type": music_event_type,
                        "pitch": pitch,
                        "duration": duration,
                        "tempo": find_tempo_at_music_event_start(music_event_start, tempo_changes),
                        "instrument": instrument_name,
                        "part": part_number,
                        "measure": measure_number,
                        "position_in_measure": position_in_measure,
                        "event_start": music_event_start
                    }
                )

        music_events = sorted(music_events, key=lambda music_event: (music_event["event_start"], music_event["part"]))
        previous_event_start = None

        for music_event in music_events:
            current_event_start = music_event["event_start"]

            if previous_event_start is None:
                time_shift = 0.0
            else:
                time_shift = find_nearest_allowed_value(max(0.0, current_event_start - previous_event_start), TIME_SHIFTS)

            music_event["time_shift"] = time_shift
            del music_event["event_start"]

            previous_event_start = current_event_start

    except Exception as e:
        print(f"Greska u {file_path}: {e}")

    return music_events

def parse_midi_folder(midi_folder, parsed_folder, dataset_name):
    print(f"Parsiranje {dataset_name} MIDI fajlova")

    if not os.path.exists(midi_folder):
        print(f"Folder sa midi fajlovima ne postoji: {midi_folder}")
        return

    files = sorted(f for f in os.listdir(midi_folder) if f.endswith('.mid'))
    print(f"Pronadjeno {len(files)} fajlova u folderu preuzedih midi fajlova")

    os.makedirs(parsed_folder, exist_ok=True)

    for file_name in files:
        midi_path = os.path.join(midi_folder, file_name)
        json_name = file_name.replace('.mid', '.json')
        json_path = os.path.join(parsed_folder, json_name)

        if os.path.exists(json_path):
            print(f"Preskocena kompozicija posto vec postoji preuzeta: {json_name}")
            continue

        music_events = parse_midi_file(midi_path)

        if len(music_events) == 0:
            print(f"Ova kompozicija nema muzickih dogadjaja: {file_name}")
            continue

        with open(json_path, 'w') as f:
            json.dump(music_events, f)

        print(f"Sacuvano {json_name} {len(music_events)} muzickih dogadjaja")

    print(f"Zavrsen kompozitor: {dataset_name}")

if __name__ == "__main__":
    parse_midi_folder(
        midi_folder="data/bach/completed",
        parsed_folder="data/parsed/bach/completed",
        dataset_name="Bach completed"
    )

    parse_midi_folder(
        midi_folder="data/bach/unfinished",
        parsed_folder="data/parsed/bach/unfinished",
        dataset_name="Bach unfinished"
    )

    parse_midi_folder(
        midi_folder="data/schubert/completed",
        parsed_folder="data/parsed/schubert/completed",
        dataset_name="Schubert completed"
    )

    parse_midi_folder(
        midi_folder="data/schubert/unfinished",
        parsed_folder="data/parsed/schubert/unfinished",
        dataset_name="Schubert unfinished"
    )

    print("Parsiranje kompletno")