# ============================
# 🎵 Full Audio Preprocessing Script (with Total Duration Reporting)
# ============================

import os
import soundfile as sf
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# === CONFIGURATION ===
folder_name = "Chopin"
RAW_DATA_DIR = Path(f"data_raw/{folder_name}")
PROCESSED_DIR = Path(f"data_processed/{folder_name}")
# TARGET_SR = 24000
TARGET_SR = 48000
CLIP_DURATION = 60  # seconds
TARGET_RMS = 0.1

# === Create output directory ===
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === Duration accumulator (global) ===
total_audio_seconds = 0  # we'll update this as clips are saved

def convert_to_wav_48khz_stereo(in_path: Path, out_path: Path):
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(2)
    audio.export(out_path, format="wav")

def rms_normalize(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    return audio * (target_rms / rms)

def is_silent(clip: np.ndarray, silence_threshold_db: float = -45.0) -> bool:
    rms = np.sqrt(np.mean(clip ** 2))
    if rms < 1e-10:
        return True
    db = 20 * np.log10(rms)
    return db < silence_threshold_db

def split_into_clips(wav_path: Path, output_dir: Path, log_path: Path = None):
    global total_audio_seconds

    y, sr = sf.read(wav_path)
    #if len(y.shape) > 1:
    #    y = np.mean(y, axis=1)
    assert sr == TARGET_SR, f"Expected sample rate {TARGET_SR}, got {sr}"

    # 🔊 OPTIONAL: Apply global RMS normalization (commented out to preserve dynamics)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-6:
         y = y * (TARGET_RMS / rms)
    #------------------------------------------------
    samples_per_clip = CLIP_DURATION * sr
    total_clips = len(y) // samples_per_clip

    silent_clips = []
    non_silent_found = False

    for i in range(total_clips):
        clip = y[i * samples_per_clip : (i + 1) * samples_per_clip]
        if is_silent(clip):
            silent_clips.append(f"{wav_path.name} [segment {i}]")
            continue
        non_silent_found = True

        fade_len = int(0.02 * sr)
        if len(clip) >= 2 * fade_len:
            fade_in = np.linspace(0, 1, fade_len).reshape(-1, 1)
            fade_out = np.linspace(1, 0, fade_len).reshape(-1, 1)
            clip[:fade_len] *= fade_in
            clip[-fade_len:] *= fade_out


        clip_name = wav_path.stem + f"_{i:03d}.wav"
        clip_path = output_dir / clip_name
        sf.write(clip_path, clip, samplerate=sr)
        total_audio_seconds += CLIP_DURATION

    if log_path and silent_clips:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            for entry in silent_clips:
                f.write(f"SILENT: {entry}\n")

    if not non_silent_found:
        wav_path.unlink(missing_ok=True)
        print(f"🗑️ Deleted silent file: {wav_path}")

    for clip_file in output_dir.glob(wav_path.stem + "_*.wav"):
        try:
            clip_data, clip_sr = sf.read(clip_file)
            if len(clip_data) < CLIP_DURATION * clip_sr * 0.5:
                clip_file.unlink()
                print(f"🗑️ Deleted short clip: {clip_file.name}")
        except Exception as e:
            print(f"⚠️ Could not check {clip_file}: {e}")


def preprocess_all_audio():
    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    audio_files = [p for p in RAW_DATA_DIR.rglob("*") if p.suffix.lower() in audio_extensions]

    print(f"Found {len(audio_files)} audio files.")

    for in_file in tqdm(audio_files):
        rel_path = in_file.relative_to(RAW_DATA_DIR)
        clean_wav_path = PROCESSED_DIR / rel_path.with_suffix(".wav")
        clean_wav_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            convert_to_wav_48khz_stereo(in_file, clean_wav_path)
            split_into_clips(clean_wav_path, clean_wav_path.parent)
            clean_wav_path.unlink()
        except Exception as e:
            print(f"❌ Failed to process {in_file}: {e}")

    # Final duration summary
    hours = int(total_audio_seconds // 3600)
    minutes = int((total_audio_seconds % 3600) // 60)
    seconds = int(total_audio_seconds % 60)
    print(f"\n🎧 Total processed audio duration: {hours}h {minutes}m {seconds}s ({total_audio_seconds:.1f} seconds)")


# === Script Entry Point ===
if __name__ == '__main__':
    preprocess_all_audio()
