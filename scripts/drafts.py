#===============================================================================================#
# 1. List all instruments in the MIDI file
midi = pretty_midi.PrettyMIDI("Chopin_midi_files_split/24Winter_wind_etude._Sequenced_by_Robert_Finley.mid")
for instrument in midi.instruments:
    print(f"Instrument: {instrument.name}, Program: {instrument.program}, Is Drum: {instrument.is_drum}")

# 2. Print all notes for each instrument
for instrument in midi.instruments:
    print(f"\nInstrument: {instrument.name}")
    for note in instrument.notes:
        print(f"Pitch: {note.pitch}, Start: {note.start:.2f}s, End: {note.end:.2f}s, Velocity: {note.velocity}")

# 3. Get tempo changes
tempos, tempo_times = midi.get_tempo_changes()
for t, time in zip(tempos, tempo_times):
    print(f"Tempo: {t} BPM at {time:.2f} seconds")

# 4. Get time signature changes
for ts in midi.time_signature_changes:
    print(f"Time Signature: {ts.numerator}/{ts.denominator} at {ts.time:.2f} seconds")

# 5. Get key signature changes
for ks in midi.key_signature_changes:
    print(f"Key Signature: {ks.key_number} at {ks.time:.2f} seconds")

#===============================================================================================#




import os
import pretty_midi

def is_midi_file_valid(file_path):
    try:
        midi = pretty_midi.PrettyMIDI(file_path)

        # Must have at least one instrument
        if len(midi.instruments) == 0:
            return False, "No instruments"

        # All instruments empty?
        if all(len(instr.notes) == 0 for instr in midi.instruments):
            return False, "All instruments are empty"

        # All tempos 0?
        tempos, _ = midi.get_tempo_changes()
        if all(t == 0.0 for t in tempos):
            return False, "All tempos are 0.0"

        # Duration too short or too long?
        duration = midi.get_end_time()
        if duration < 1.0:
            return False, "File too short"
        if duration > 3600:
            return False, "File too long (>1 hour)"

        return True, "Valid MIDI"

    except Exception as e:
        return False, f"Corrupted or unreadable: {e}"

# === Batch scan the folder
folder = "/home/zhengmy/ECS111_FP/Chopin_midi_files_split"
for filename in os.listdir(folder):
    if filename.lower().endswith(".mid"):
        path = os.path.join(folder, filename)
        is_valid, reason = is_midi_file_valid(path)
        print(f"{filename}: {'✅ VALID' if is_valid else '❌ INVALID'} — {reason}")
