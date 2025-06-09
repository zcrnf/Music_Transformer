#!/usr/bin/env python
"""
Generate a music clip with the multi‑codebook Transformer trained on Encodec tokens (48 kHz).
This script exactly re‑uses the architecture from training and therefore loads the
checkpoint without key/shape errors, then decodes the generated tokens to a WAV file.
"""
from pathlib import Path
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from encodec import EncodecModel
import soundfile as sf
import torchaudio            


# ───────────────────────── CONFIG ─────────────────────────

CHECKPOINT_EPOCH = int(input("enter the epochs number you want to retireve: "))
folder_name      = input("Enter the name of artist: ")
DESIRED_SECONDS  = 60
TEMPERATURE      = float(input("Enter temperature value (e.g. 0.7): "))
TOP_K            = int(input("Enter the top K value (e.g. 32): "))       # ← 保留原先 top-k
TOP_P            = float(input("Enter top-p value (e.g. 0.92): "))     # ← 新增；想用 nucleus 就填 0.95 之类的概率
repitition_penalty = float(input("Enter repitition_penalty (e.g. 1.015): "))
BOS_ID           = 1024
PAD_ID           = 1025          
CHECKPOINT_PATH  = Path(f"model_results_{folder_name}/music_transformer_ep{CHECKPOINT_EPOCH}.pt")
TOK_OUT_PATH     = Path(f"music_transformer/{folder_name}/ep{CHECKPOINT_EPOCH}.th")
WAV_OUT_PATH     = Path(f"music_transformer/{folder_name}/ep{CHECKPOINT_EPOCH}.wav")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────── MODEL (identical to training) ────────────────────
class CodebookMultiHeadTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_embed: int, Q: int,
                 d_model: int, seq_len: int,
                 n_layer: int, n_head: int):
        super().__init__()
        self.Q = Q
        self.vocab_size = vocab_size

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_embed) for _ in range(Q)
        ])
        self.linear_proj = nn.Linear(Q * d_embed, d_model)

        cfg = GPT2Config(
            vocab_size=1,
            n_positions=seq_len,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head
        )
        self.transformer = GPT2Model(cfg)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(Q)
        ])

    def forward(self, input_ids: torch.Tensor):
        B, T, Q = input_ids.shape
        assert Q == self.Q, "Mismatch in code-book dimension"

        embeds = [self.embeddings[q](input_ids[:, :, q]) for q in range(Q)]
        concat = torch.cat(embeds, dim=-1)
        inputs_embeds = self.linear_proj(concat)

        hidden = self.transformer(inputs_embeds=inputs_embeds).last_hidden_state
        logits = [head(hidden) for head in self.heads]
        return logits

# ──────────────────── LOAD CHECKPOINT ────────────────────
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

Q          = ckpt['Q']
d_embed    = ckpt['d_embed']
d_model    = ckpt['d_model']
seq_len    = ckpt['seq_len']
n_layer    = ckpt['n_layer']
n_head     = ckpt['n_head']
if 'vocab_size' in ckpt:
    vocab_size = ckpt['vocab_size']
else:
    print("⚠️ 'vocab_size' not found in checkpoint — defaulting to 1025")
    vocab_size = 1026

model = CodebookMultiHeadTransformer(
    vocab_size=vocab_size,
    d_embed=d_embed,
    Q=Q,
    d_model=d_model,
    seq_len=seq_len,
    n_layer=n_layer,
    n_head=n_head
).to(DEVICE).eval()

model.load_state_dict(ckpt['model_state_dict'], strict=True)
print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} (seq_len={seq_len})")


# ──────────────────── SAMPLING HELPERS ────────────────────
def pretty_vec(vec):
    """turn a Q-length tensor into a short string like 0-3-7-0-..."""
    return "-".join(map(str, vec.tolist()))

def top_k_sample(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sample from the top-k tokens in the logits."""
    top_vals, top_idx = torch.topk(logits, k)
    logits_filtered   = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(1, top_idx, top_vals)
    probs = torch.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def top_p_sample(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling for a single logit row (B==1)."""
    probs, idx = torch.sort(torch.softmax(logits, -1), descending=True)
    cdf = torch.cumsum(probs, -1)
    mask = cdf - probs > p        # tokens after cutoff
    probs[mask] = 0.0
    probs = probs / probs.sum(-1, keepdim=True)
    token = torch.multinomial(probs, 1)
    return idx.gather(-1, token).squeeze(-1)


# ──────────────────── TOKEN GENERATION ────────────────────
from itertools import islice
token_dir = Path(f"encoded_tokens/{folder_name}")
first_token_path = next(islice(token_dir.rglob("*.th"), 1))  # pick one file
seed_raw = torch.load(first_token_path)  # [1, Q, T]

# If shape is [1, 1, Q, T], squeeze the second dim 
if seed_raw.ndim == 4 and seed_raw.shape[1] == 1:
    seed_raw = seed_raw.squeeze(1)  # → [1, Q, T]

# Final check: must be [1, Q, T]
if seed_raw.ndim != 3 or seed_raw.shape[0] != 1:
    raise ValueError(f"❌ Unexpected shape for seed tokens: {seed_raw.shape}")

# Transpose to [1, T, Q] for Transformer input
seed_tokens = seed_raw.squeeze(0).transpose(0, 1).unsqueeze(0).to(DEVICE)

# Truncate to SEED_SECONDS
TOKENS_PER_SEC = 48000 // 320
SEED_SECONDS = int(input("Enter the #seconds you want as a seed: "))
TOKENS_TO_GEN = DESIRED_SECONDS * TOKENS_PER_SEC
# ─── entropy‐triggered temperature boost ────────────────────────────────
ENTROPY_TRIGGER = 6.0            # bits threshold for 1-sec window
TEMP_BOOST      = float(input("Enter the temperature boost when bad output is found(e.g. 1.5): "))            # factor to multiply TEMPERATURE by
WINDOW          = TOKENS_PER_SEC # frames in 1 second

def window_entropy(seq_1d):
    # compute Shannon entropy over last WINDOW frames of 1D token stream
    ids, counts = torch.unique(seq_1d[-WINDOW:], return_counts=True)
    p = counts.float() / counts.sum()
    return -(p * torch.log2(p + 1e-12)).sum()

# seed_tokens = seed_tokens[:, :SEED_SECONDS * TOKENS_PER_SEC]
seed_tokens = seed_tokens[:, :min(seed_tokens.shape[1], SEED_SECONDS * TOKENS_PER_SEC)]

bos = torch.full((1, 1, Q), BOS_ID, dtype=torch.long, device=DEVICE)
generated = torch.cat([bos, seed_tokens], dim=1)      # prepend BOS

print(f"🔁 Using warm-up seed from {first_token_path.name}, shape: {seed_tokens.shape}")
# generated = seed_tokens  # [1, T, Q]


#generated = torch.full((1, 1, Q), BOS_ID, dtype=torch.long, device=DEVICE)  # [B,1,Q]

def apply_scaled_repetition_penalty(logits, generated_tokens, penalty=1.015):
    # Validate input dimensions
    vocab_size = logits.size(-1)
    if generated_tokens.max() >= vocab_size:
        raise ValueError(f"⚠️ Token ID {generated_tokens.max().item()} exceeds vocab size {vocab_size}")

    # Compute frequency-based penalty scaling
    counts = torch.bincount(generated_tokens, minlength=vocab_size).float()
    scaling = penalty ** counts
    logits = logits / scaling.unsqueeze(0)

    # Sanitize logits
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return logits


with torch.no_grad():
    step_idx  = generated.shape[1]
    step_temp = TEMPERATURE

    while step_idx < TOKENS_TO_GEN:
        # 1) get logits
        context     = generated[:, -seq_len:]
        logits_list = model(context)

        # 2) only every WINDOW steps recompute collapse & temp
        if step_idx % WINDOW == 0:
            entropies  = [window_entropy(generated[0, :, q]) for q in range(Q)]
            n_collapse = sum(H < ENTROPY_TRIGGER for H in entropies)
            if n_collapse >= 0.7 * Q:
                step_temp = TEMPERATURE * TEMP_BOOST
                print(f"⚡ Boost ALL heads ({n_collapse}/{Q} collapsed)")
            else:
                step_temp = TEMPERATURE

        # 3) slice out last‐token logits
        last_logits = [lg[:, -1] for lg in logits_list]

        # 4) sample each head
        next_tokens = []
        for q, logit_q in enumerate(last_logits):
            # 4a) repetition penalty
            logit_q = apply_scaled_repetition_penalty(
                logit_q, generated[0, :, q], repitition_penalty
            )
            # 4b) apply the (possibly boosted) temperature
            logit_q = logit_q / step_temp

            # 4c) hard masks
            logit_q[:, BOS_ID] = -float("inf")
            logit_q[:, PAD_ID] = -float("inf")

            # 4d) top-k / top-p
            if TOP_K and TOP_K > 0:
                topv, topi = torch.topk(logit_q, TOP_K, dim=-1)
                filt = torch.full_like(logit_q, -float("inf"))
                filt.scatter_(1, topi, topv)
                logit_q = filt

            if TOP_P:
                next_tok = top_p_sample(logit_q, TOP_P)
            else:
                probs    = torch.softmax(logit_q, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)

            next_tokens.append(next_tok)

        # 5) append and advance
        generated = torch.cat(
            [generated,
             torch.stack(next_tokens, dim=-1).unsqueeze(1)],
            dim=1
        )
        step_idx += 1


                
flat = generated.squeeze(0)              # [T, Q]
print("unique token IDs (full clip):", torch.unique(flat))
print(f"📝 Generated token sequence shape: {generated.shape}")

tokens = generated.squeeze(0)                  # [T,Q]
tokens_no_bos = tokens[1:].T.unsqueeze(0)      # [1,Q,T]

# ── per-second token diagnostics ────────────────────────────
print("\n🩺  Per-second token summary (first frame + diversity):")
for sec in range(DESIRED_SECONDS):
    start = sec * TOKENS_PER_SEC
    end   = start + TOKENS_PER_SEC
    if end > generated.shape[1] - 1:       # safety
        break
    window = generated[0, start:end]       # [Tsec, Q]
    first_vec = window[0]
    unique_qvecs = torch.unique(window, dim=0)
    mode_vec, mode_cnt = torch.mode(window, dim=0)
    print(f"  sec {sec:02d}:  first={pretty_vec(first_vec)}  "
          f"unique={len(unique_qvecs):3d}  "
          f"mode={pretty_vec(mode_vec)} ({mode_cnt.sum().item()} / {TOKENS_PER_SEC})")
print("─────────────────────────────────────────────────────────\n")

# Skip first 3 seconds of generated tokens to remove startup noise
#SKIP = 3 * TOKENS_PER_SEC
#tokens_trim = generated.squeeze(0)[SKIP:]       # shape [T-450, Q]
#tokens_no_bos = tokens_trim.T.unsqueeze(0)      # shape [1, Q, T-450]

# Sanity check: make sure BOS is gone and all tokens are in 0-1023 range
uniq_decoding = torch.unique(tokens_no_bos)
print("🔍 Unique IDs for decoding (after BOS strip):", uniq_decoding)
if (uniq_decoding >= 1024).any():
    print("⚠️ Warning: Some invalid tokens (≥1024) still present!")
else:
    print("✅ All decoding tokens within expected range (0-1023)")

TOK_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(tokens_no_bos, TOK_OUT_PATH)
print(f"💾 Saved tokens → {TOK_OUT_PATH}")

#uniq, counts = torch.unique(flat[:900], return_counts=True)
#print({int(u): int(c) for u, c in zip(uniq[:20], counts[:20])})

# ──────────────────── DECODE TO AUDIO ────────────────────
codec = EncodecModel.encodec_model_48khz().to(DEVICE).eval()
with torch.no_grad():
    # decoded = codec.decode([(tokens_no_bos.to(DEVICE), None)])[0]
    dummy_scale = torch.ones(1, Q, 1, device=DEVICE)
    decoded = codec.decode([(tokens_no_bos.to(DEVICE), dummy_scale)])


wav = decoded[0].squeeze().cpu().numpy().astype("float32")
wav = (wav / max(1e-6, abs(wav).max())).clip(-1 + 1e-6, 1 - 1e-6)
wav = wav.T                                           # (T,2)

WAV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(WAV_OUT_PATH.as_posix(), wav, 48000, subtype="PCM_16")

# slowdown (optional,与原脚本一致)
# wav_tensor, sr = torchaudio.load(WAV_OUT_PATH)
# resampler   = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr)  # no change
# slowed_wav  = resampler(wav_tensor)
# SLOWED_WAV_PATH = WAV_OUT_PATH.with_name(WAV_OUT_PATH.stem + "_slowed.wav")
# sf.write(SLOWED_WAV_PATH.as_posix(), slowed_wav.squeeze(0).numpy().T, sr)

print(f"🎧 Wrote {DESIRED_SECONDS}s clip to → {WAV_OUT_PATH}")




import matplotlib.pyplot as plt

# Safely move to CPU for plotting (this is required)
uniq, counts = torch.unique(tokens_no_bos, return_counts=True)
uniq_cpu = uniq.detach().cpu()
counts_cpu = counts.detach().cpu()

plt.bar(uniq_cpu, counts_cpu)
plt.title("Token Frequency Distribution")
plt.xlabel("Token ID")
plt.ylabel("Count")
plt.show()
