import pretty_midi
from pathlib import Path
import os
import numpy as np
import re

name = 'Chopin'
# === Configuration ===
midi_dir = Path(f"{name}_midi_files")               # Your original MIDI folder
output_dir = Path(f"{name}_midi_files_split")        # New folder for safe files
output_dir.mkdir(exist_ok=True)

MIN_DURATION = 60    # Minimum 1.5 minutes
MAX_DURATION = 240   # Maximum 5 minutes

# Data tracking
durations = []
valid_filenames = []
corrupt_filenames = []


# === Filename sanitizer to avoid Google Drive issues ===
def safe_filename(name):
    name = re.sub(r'[^\x00-\x7F]+', '', name)
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(" ", "_").replace(",", "").replace("’", "")
    name = name.replace("(", "").replace(")", "")
    return name

# === Function to process MIDIs ===
def process_midi(midi_path, min_duration=MIN_DURATION, max_duration=MAX_DURATION):
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        total_time = midi.get_end_time()

        # Filter: skip files too short
        if total_time < min_duration:
            print(f"⚠️ Skipped (too short): {midi_path.name} ({total_time:.2f} sec)")
            return

        safe_stem = safe_filename(midi_path.stem)

        if total_time <= max_duration:
            # Short enough: just copy
            new_path = output_dir / f"{safe_stem}.mid"
            midi.write(str(new_path))
            durations.append(total_time)
            valid_filenames.append(new_path.name)
            print(f"✅ Copied: {new_path.name} ({total_time:.2f} sec)")

        else:
            # Split into chunks until the last chunk is <90s and discard it
            print(f"✂️ Splitting {midi_path.name} ({total_time:.2f} sec)")
            start = 0
            part_idx = 1
            while start + min_duration <= total_time:
                end = min(start + max_duration, total_time)
                if (end - start) < min_duration:
                    break  # remaining piece too short, discard

                new_midi = pretty_midi.PrettyMIDI()

                for instrument in midi.instruments:
                    new_instrument = pretty_midi.Instrument(
                        program=instrument.program,
                        is_drum=instrument.is_drum,
                        name=instrument.name
                    )
                    new_instrument.notes = [
                        note for note in instrument.notes if start <= note.start < end
                    ]
                    if new_instrument.notes:
                        new_midi.instruments.append(new_instrument)

                # Shift notes
                for instrument in new_midi.instruments:
                    for note in instrument.notes:
                        note.start -= start
                        note.end   -= start

                # Save split file
                split_filename = f"{safe_stem}_part{part_idx}.mid"
                split_path = output_dir / split_filename
                new_midi.write(str(split_path))

                durations.append(end - start)
                valid_filenames.append(split_filename)
                print(f"✅ Saved split: {split_filename} ({end-start:.2f} sec)")

                start += max_duration
                part_idx += 1

    except Exception as e:
        print(f"❌ Failed on {midi_path.name}: {e}")
        corrupt_filenames.append(midi_path)

# === Main Loop: process all MIDIs ===
for midi_path in midi_dir.glob("*.mid"):
    process_midi(midi_path)

# === Delete corrupted files ===
for bad_file in corrupt_filenames:
    try:
        os.remove(bad_file)
        print(f"🗑️ Removed corrupted file: {bad_file.name}")
    except Exception as e:
        print(f"⚠️ Failed to delete {bad_file.name}: {e}")

# === Summary statistics ===
if durations:
    durations = np.array(durations)
    print("\n=== Summary Statistics After Cleaning ===")
    print(f"✅ Total valid MIDI files: {len(valid_filenames)}")
    print(f"⏱️ Total music time: {durations.sum() / 3600:.2f} hours")
    print(f"📏 Min duration: {durations.min():.2f} sec")
    print(f"📏 Max duration: {durations.max():.2f} sec")
    print(f"📏 Mean duration: {durations.mean():.2f} sec")
    print(f"📏 Median duration: {np.median(durations):.2f} sec")
else:
    print("\n⚠️ No valid MIDI files found.")
