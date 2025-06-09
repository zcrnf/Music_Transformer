from pathlib import Path

def list_training_audio(raw_dir="data_raw", audio_exts={".mp3", ".wav", ".flac", ".ogg", ".m4a"}):
    raw_path = Path(raw_dir)
    audio_files = sorted([str(p.relative_to(raw_path)) for p in raw_path.rglob("*") if p.suffix.lower() in audio_exts])
    
    print(f"🔍 Found {len(audio_files)} audio files in '{raw_dir}'")
    for i, fname in enumerate(audio_files):
        print(f"[{i+1}] {fname}")
    
    return audio_files

all_files = list_training_audio()



import os
from pathlib import Path

# Configuration
folder_name = "Chopin"  # You can change this dynamically
RAW_DATA_DIR = Path(f"data_raw/{folder_name}")
audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Gather and print all raw file names
raw_file_names = [p.name for p in RAW_DATA_DIR.rglob("*") if p.suffix.lower() in audio_extensions]

print(f"Found {len(raw_file_names)} raw audio files:")
for name in raw_file_names:
    print(name)



from pathlib import Path
import shutil

# Set paths
RAW_DIR = Path("data_raw")
OUT_DIR = Path("data_filtered_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define audio file extensions
AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}

# Define keywords that suggest presence of string instruments (case-insensitive)
STRING_KEYWORDS = {
    "cello", "violin", "viola", "double bass", "contrabass", "string", "orchestra",
    "trio", "duo", "ensemble", "concertgebouw", "quartet", "quintet", "sextet", "septet",
    "kondrashin", "filharmonia", "symphonique", "piano concerto", "concerto",
    "andante spianato et grande polonaise", "introduction and polonaise",
    "introduction and variations", "rondo for two pianos"
}

# Move matching files to OUT_DIR
moved = 0
for file in RAW_DIR.rglob("*"):
    if not file.is_file() or file.suffix.lower() not in AUDIO_EXTENSIONS:
        continue

    name = file.name.lower()
    if any(keyword in name for keyword in STRING_KEYWORDS):
        dest = OUT_DIR / file.relative_to(RAW_DIR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(dest))
        print(f"🚚 Moved: {file.relative_to(RAW_DIR)} → {dest.relative_to(OUT_DIR)}")
        moved += 1

print(f"\n✅ Done. Moved {moved} string-instrument files to '{OUT_DIR}/'")





from pathlib import Path
import shutil

# Set base directories
RAW_DIR = Path("data_raw")
OUT_DIR = Path("data_filtered_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filename to search for
target_name = "Arthur Moreira Lima,Филхармония България,Frédéric François Chopin - Krakowiak in F major , Op. 14.flac"

# Search for file recursively
matches = list(RAW_DIR.rglob(target_name))

if not matches:
    print(f"❌ File not found anywhere under {RAW_DIR}")
else:
    for match in matches:
        relative_path = match.relative_to(RAW_DIR)
        dest_path = OUT_DIR / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(match), str(dest_path))
        print(f"✅ Moved: {match} → {dest_path}")
