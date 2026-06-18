# 2_Tokenization.py — FINAL CLEAN FIXED version for tokenization
import json
from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile

NAME = "Chopin"  # or whatever your corpus is called

MIDI_DIR = Path(f"{NAME}_midi_files_split")
TOK_DIR = Path(f"{NAME}_midi_tokens")
TOK_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Set up full vocabulary ───────────────────────────────────────────────
cfg = TokenizerConfig(
    pitch_range=(21, 108),              # full piano range A0 to C8
    beat_res={(0, 4): 4, (4, 8): 8},     # two beat resolutions
    use_programs=True,
    use_tempos=True,
    num_tempos=32,                      # corrected name
    num_velocities=127,                 # corrected name
    tempo_range=(30, 240),
    additional_tokens={
        "Pad": True,
        "BOS": True,
        "EOS": True
    }
)

tokenizer = REMI(cfg)

# ─── 2. Clean and save TokenizerConfig safely ─────────────────────────────────
def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [clean_json(x) for x in obj]
    elif isinstance(obj, (int, float, str)) or obj is None:
        return obj
    else:
        return str(obj)  # fallback for unknown types

TOKENIZER_JSON = Path("tokenizer_config.json")
cfg_dict = clean_json(cfg.to_dict())

with TOKENIZER_JSON.open("w", encoding="utf-8") as f:
    json.dump(cfg_dict, f, indent=2)

# ─── 3. Tokenize all MIDI files ──────────────────────────────────────────────
META_PATH = Path("metadata_midi.jsonl")
with META_PATH.open("w", encoding="utf-8") as meta_f:
    for midi_path in MIDI_DIR.glob("*.mid"):
        try:
            tokens = tokenizer(MidiFile(str(midi_path)))
            token_out = TOK_DIR / (midi_path.stem + ".json")
            tokenizer.save_tokens(tokens, token_out)

            meta_f.write(json.dumps({
                "midi": str(midi_path),
                "tokens": str(token_out),
                "text": f"{NAME}-style piece"
            }) + "\n")
            print(f"✅ tokenized {midi_path.name}")
        except Exception as e:
            print(f"❌ Failed on {midi_path.name}: {e}")

print(f"\n🎉 Done: vocabulary size = {tokenizer.vocab_size}")
