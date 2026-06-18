# 4_MIDI_Inference.py  — manual fallback for very-old miditok
import json, pathlib, torch, pretty_midi
from transformers import GPT2LMHeadModel
from miditok import REMI, TokSequence, TokenizerConfig

MODEL_DIR      = "saved_model_epochs200_fixed"
TOKENIZER_JSON = "tokenizer_config.json"
OUT_MIDI       = "generated_output.mid"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 1️⃣  model
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device).eval()

# 2️⃣  tokenizer
with open(TOKENIZER_JSON) as f:
    cfg = TokenizerConfig(**json.load(f))
tokenizer = REMI(cfg)

# 3️⃣  prompt (single BOS 0)
prompt = torch.tensor([[0]], dtype=torch.long, device=device)

# 4️⃣  generate
gen = model.generate(
    prompt,
    max_new_tokens=1024,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    pad_token_id=model.config.eos_token_id,
)
ids = gen[0].tolist()[1:]                 # drop BOS

# 5️⃣  old API: tokens → Symusic ScoreTick
score = tokenizer.tokens_to_midi([ids])
if isinstance(score, list):               # some versions return a list
    score = score[0]

# 6️⃣  manual ScoreTick ➜ PrettyMIDI
def scoretick_to_pretty_midi(scr):
    pm = pretty_midi.PrettyMIDI()
    for track in getattr(scr, "tracks", []):
        prog    = int(getattr(track, "program", 0))
        is_drum = bool(getattr(track, "is_drum", False))
        inst    = pretty_midi.Instrument(program=prog, is_drum=is_drum)
        for note in getattr(track, "notes", []):
            pitch = int(getattr(note, "pitch", 60))
            vel   = int(getattr(note, "velocity", 80))
            start = float(getattr(note, "start", 0.0))
            end   = float(getattr(note, "end", start + 0.5))
            inst.notes.append(pretty_midi.Note(vel, pitch, start, end))
        pm.instruments.append(inst)
    return pm

print("Converting (manual fallback) …")
pm = scoretick_to_pretty_midi(score)
pm.write(OUT_MIDI)
print(f"✅  MIDI saved → {pathlib.Path(OUT_MIDI).resolve()}")
