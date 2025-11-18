"""
MusicGen Metadata Preparation Script (No Artist/Album Prompts)

This script:
1. Scans all .th token files in 'encoded_tokens/' and writes a `metadata.csv`
2. Converts it to `metadata.jsonl`, with duration/sample rate for each clip

All prompts are empty ("") for unconditional training.
"""

import os
import csv
import json
import soundfile as sf
from pathlib import Path

# === CONFIG ===
folder_name = "Chopin"
TOKEN_DIR = Path(f"encoded_tokens/{folder_name}")  # No artist subdir required
CSV_PATH = f"metadata_{folder_name}.csv"
JSONL_PATH = f"metadata_{folder_name}.jsonl"
JSONL_CLEAN = f"metadata_clean_{folder_name}.jsonl"

# === Main Metadata Generator ===
def generate_metadata_from_folder_structure():
    th_files = sorted(TOKEN_DIR.rglob("*.th"))
    count = 0

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file, open(JSONL_PATH, "w", encoding="utf-8") as jsonl_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath", "text"])

        for th_file in th_files:
            try:
                relative = th_file.relative_to(TOKEN_DIR)  # keep subfolder structure if any
                token_path = (TOKEN_DIR / relative).as_posix()
                prompt = ""  # ← No text conditioning

                # Try to find matching .wav for metadata (duration/sample rate)
                audio_path = Path(f"data_processed/{folder_name}") / relative.with_suffix(".wav")
                try:
                    f = sf.SoundFile(audio_path)
                    duration = len(f) / f.samplerate
                    sample_rate = f.samplerate
                except:
                    duration = None
                    sample_rate = None

                writer.writerow([token_path, prompt])
                json.dump({
                    "audio": token_path,
                    "text": prompt,
                    "path": token_path,
                    "duration": duration,
                    "sample_rate": sample_rate
                }, jsonl_file)
                jsonl_file.write("\n")
                count += 1

            except Exception as e:
                print(f"❌ Error with {th_file}: {e}")

    print(f"✅ Wrote metadata for {count} files.")

# === Optional JSONL cleaner ===
def clean_jsonl(input_path, output_path):
    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                obj = json.loads(line.strip())
                obj["audio"] = Path(obj["audio"]).as_posix()
                obj["path"] = Path(obj["path"]).as_posix()
                json.dump(obj, outfile)
                outfile.write("\n")
                count += 1
            except Exception:
                continue
    
    print(f"✅ Cleaned JSONL written: {count} entries → {output_path}")

# === Run everything ===
if __name__ == "__main__":
    generate_metadata_from_folder_structure()
    clean_jsonl(JSONL_PATH, JSONL_CLEAN)




