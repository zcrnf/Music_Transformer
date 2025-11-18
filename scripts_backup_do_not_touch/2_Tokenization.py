# === Tokenization Script with MP3 Support ===
# Author: You
# Description: Tokenizes all 10s audio clips (mp3/wav/flac/etc) from `data_processed/` using Encodec

import os
from pathlib import Path
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

# === CONFIG ===
folder_name = "Chopin"
DATA_DIR = Path(f"data_processed/{folder_name}")             # Folder containing mp3/wav/etc
OUTPUT_DIR = Path(f"encoded_tokens/{folder_name}")           # Token file output
SAMPLE_RATE = 48000                           
BANDWIDTH = 12.0                              # Target bitrate (e.g., 6.0 → 8 codebooks)

def tokenize_dataset():
    #model = EncodecModel.encodec_model_24khz()
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(BANDWIDTH)
    model.eval()

    print("🚀 Starting tokenization...")

    audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = sorted([f for f in DATA_DIR.rglob("*") if f.suffix.lower() in audio_exts])
    total_duration_sec = 0.0

    for audio_path in audio_files:
        relative_path = audio_path.relative_to(DATA_DIR)
        token_path = OUTPUT_DIR / relative_path.with_suffix(".th")
        token_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # === Load and convert audio to 24kHz mono ===
            wav, sr = torchaudio.load(audio_path)
            duration_sec = wav.shape[1] / sr
            total_duration_sec += duration_sec  # add to total

            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav = wav.unsqueeze(0)  # [1, 2, T] for stereo

            # === Encode with Encodec ===
            with torch.no_grad():
                encoded = model.encode(wav)

                # Unpack tuple (codes, scale)
                if isinstance(encoded, tuple):
                    encoded = encoded[0]

                # If list of tuples (per-chunk outputs), extract and stack
                if isinstance(encoded, list):
                    if isinstance(encoded[0], tuple):
                        encoded = [e[0] for e in encoded]
                    encoded = torch.cat(encoded, dim=2)  # [B, Q, T] or [1, 1, Q, T]

                # Squeez the shape of [1, 1, Q, T] to [1, Q, T]
                if isinstance(encoded, torch.Tensor) and encoded.ndim == 4 and encoded.shape[1] == 1:
                    encoded = encoded.squeeze(1)

                # Final shape check
                if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3:
                    raise ValueError(f"Unexpected encoding shape: {getattr(encoded, 'shape', 'N/A')}")

                torch.save(encoded, token_path)
                print(f"✅ Tokenized: {relative_path} → shape {tuple(encoded.shape)}")

        except Exception as e:
            print(f"❌ Failed: {relative_path} — {e}")

    print(f"\n🎉 Done! All tokens saved to: {OUTPUT_DIR.resolve()}")

    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = int(total_duration_sec % 60)
    print(f"📏 Total audio duration: {hours}h {minutes}m {seconds}s ({total_duration_sec:.1f} seconds)")

# === Entry Point ===
if __name__ == "__main__":
    tokenize_dataset()


