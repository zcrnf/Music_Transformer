#!/usr/bin/env python
"""
Generate a music clip with the multi-codebook Transformer trained on Encodec tokens (48 kHz).
This version:
- Matches the upgraded trainer (padding-aware embeddings, weight tying, PAD masking in attention).
- Loads checkpoints saved with torch.compile / DDP by stripping "_orig_mod." / "module." prefixes.
- Robustly loads seed tokens (dict or tensor .th) and reuses/constructs an Encodec scale.
- Uses top-k / top-p sampling, repetition penalty (ignoring BOS/PAD), and entropy-triggered temp boost.
- **Chunked decoding** with Encodec using a CORRECT scale shape [1, 1, T], with GPU→CPU fallback to avoid OOM.
"""

from pathlib import Path
from itertools import islice
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from encodec import EncodecModel
import soundfile as sf

# ───────────────────────── USER CONFIG ─────────────────────────
CHECKPOINT_EPOCH   = int(input("enter the epochs number you want to retrieve: "))
folder_name        = input("Enter the name of artist: ")
DESIRED_SECONDS    = 60

TEMPERATURE        = float(input("Enter temperature value (e.g. 0.7): "))
TOP_K              = int(input("Enter the top K value (e.g. 32, 0 to disable): "))
TOP_P              = float(input("Enter top-p value (e.g. 0.92, 0 to disable): "))
REPETITION_PENALTY = float(input("Enter repetition_penalty (e.g. 1.015): "))
SEED_SECONDS       = int(input("Enter the #seconds you want as a seed (e.g. 4): "))
TEMP_BOOST         = float(input("Enter the temperature boost when collapse is detected (e.g. 1.2): "))

BOS_ID           = 1024
PAD_ID           = 1025
VOCAB_SIZE       = PAD_ID + 1

CHECKPOINT_PATH  = Path(f"model_results_{folder_name}/music_transformer_ep{CHECKPOINT_EPOCH}.pt")
TOK_OUT_PATH     = Path(f"music_transformer/{folder_name}/ep{CHECKPOINT_EPOCH}.th")
WAV_OUT_PATH     = Path(f"music_transformer/{folder_name}/ep{CHECKPOINT_EPOCH}.wav")
TOKEN_DIR        = Path(f"encoded_tokens/{folder_name}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sampling dynamics
TOKENS_PER_SEC  = 150                 # Encodec(48k) ≈ 150 fps
TOKENS_TO_GEN   = DESIRED_SECONDS * TOKENS_PER_SEC
ENTROPY_TRIGGER = 6.0                 # bits threshold over 1s window
WINDOW          = TOKENS_PER_SEC      # 1s window length in frames

# Decode in chunks to avoid OOM
CHUNK_FRAMES    = 2000                # ~13.3 seconds per chunk (adjust per GPU RAM)

# ──────────────────── MODEL (identical to training) ────────────────────
class CodebookMultiHeadTransformer(nn.Module):
    def __init__(self, vocab_size, d_embed, Q, d_model, seq_len, n_layer, n_head):
        super().__init__()
        self.Q = Q
        self.vocab_size = vocab_size
        self.d_embed = d_embed

        # Per-codebook embeddings (padding_idx stabilizes grads)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_embed, padding_idx=PAD_ID) for _ in range(Q)
        ])

        # Project concatenated embeddings to model dim
        self.linear_proj = nn.Linear(Q * d_embed, d_model)

        # GPT-2 backbone (uses inputs_embeds)
        cfg = GPT2Config(
            vocab_size=1,          # dummy, we use inputs_embeds
            n_positions=seq_len,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head
        )
        self.transformer = GPT2Model(cfg)
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()

        # Heads: d_model -> d_embed, then tie to embedding matrices for logits
        self.heads = nn.ModuleList([
            nn.Linear(d_model, d_embed, bias=False) for _ in range(Q)
        ])

        self.dropout = nn.Dropout(0.1)

    def _compute_attn_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Treat a time step as padding if ANY codebook is PAD at that step.
        return (input_ids[..., 0] != PAD_ID).long()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        B, T, Q = input_ids.shape
        assert Q == self.Q, f"Mismatch in codebook dimension: got {Q}, expected {self.Q}"

        embeds = [self.embeddings[q](input_ids[:, :, q]) for q in range(Q)]  # [B,T,d] each
        concat = torch.cat(embeds, dim=-1)                 # [B, T, Q*d]
        inputs_embeds = self.linear_proj(concat)           # [B, T, d_model]
        inputs_embeds = self.dropout(inputs_embeds)

        hidden = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=self._compute_attn_mask(input_ids)
        ).last_hidden_state                                 # [B, T, d_model]

        if labels is None:
            logits = []
            for q in range(Q):
                z = self.heads[q](hidden)                          # [B, T, d_embed]
                logits_q = torch.matmul(z, self.embeddings[q].weight.T)  # [B, T, V]
                logits.append(logits_q)
            return logits

        total_loss = 0.0
        for q in range(Q):
            z = self.heads[q](hidden)
            logits_q = torch.matmul(z, self.embeddings[q].weight.T)
            loss_q = F.cross_entropy(
                logits_q.reshape(-1, self.vocab_size),
                labels[:, :, q].reshape(-1),
                ignore_index=-100,
                label_smoothing=0.1
            )
            total_loss = total_loss + loss_q
        return total_loss / Q

# ──────────────────── LOAD CHECKPOINT (with prefix strip) ────────────────────
def strip_prefixes(state_dict: dict) -> dict:
    """Remove wrappers like '_orig_mod.' (torch.compile) and 'module.' (DDP)."""
    new = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        new[k] = v
    return new

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
Q          = ckpt['Q']
d_embed    = ckpt['d_embed']
d_model    = ckpt['d_model']
seq_len    = ckpt['seq_len']
n_layer    = ckpt['n_layer']
n_head     = ckpt['n_head']
vocab_size = ckpt.get('vocab_size', VOCAB_SIZE)

model = CodebookMultiHeadTransformer(
    vocab_size=vocab_size,
    d_embed=d_embed,
    Q=Q,
    d_model=d_model,
    seq_len=seq_len,
    n_layer=n_layer,
    n_head=n_head
).to(DEVICE).eval()

state = strip_prefixes(ckpt['model_state_dict'])
model.load_state_dict(state, strict=True)
print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} (seq_len={seq_len})")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ──────────────────── SAMPLING HELPERS ────────────────────
def top_p_sample(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling for a single row (B=1). Expects logits shape [1, V]."""
    probs = torch.softmax(logits, dim=-1)        # [1, V]
    probs, idx = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(probs, dim=-1)
    mask = cdf > p
    mask[..., 0] = False  # always keep the top token
    probs[mask] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    token = torch.multinomial(probs, 1)          # [1,1]
    return idx.gather(-1, token).squeeze(-1)     # [1]

def apply_scaled_repetition_penalty(logits, generated_tokens, penalty=1.015):
    """Exclude BOS/PAD from the repetition penalty; generated_tokens is 1D [T]."""
    vocab_size = logits.size(-1)
    valid = (generated_tokens >= 0) & (generated_tokens < 1024)  # only real codes
    ids = generated_tokens[valid]
    counts = torch.bincount(ids, minlength=vocab_size).float()   # [V]
    scaling = penalty ** counts
    logits = logits / scaling.unsqueeze(0)
    return torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

def window_entropy(seq_1d: torch.Tensor) -> torch.Tensor:
    # Shannon entropy over last WINDOW frames of a 1D token stream
    if seq_1d.numel() == 0:
        return torch.tensor(0.0)
    tail = seq_1d[-WINDOW:]
    ids, counts = torch.unique(tail, return_counts=True)
    p = counts.float() / counts.sum()
    return -(p * torch.log2(p + 1e-12)).sum()

# ──────────────────── TOKEN GENERATION ────────────────────
# Load one seed clip (codes and, if available, scale) from the dataset
try:
    first_token_path = next(islice(TOKEN_DIR.rglob("*.th"), 1))
except StopIteration:
    raise FileNotFoundError(f"No .th files under {TOKEN_DIR}")

seed_obj   = torch.load(first_token_path, map_location="cpu")
seed_codes = seed_obj["codes"] if isinstance(seed_obj, dict) else seed_obj  # [1,Q,T]
seed_scale = seed_obj.get("scale") if isinstance(seed_obj, dict) else None

# normalize shapes
if seed_codes.ndim == 4 and seed_codes.shape[1] == 1:
    seed_codes = seed_codes.squeeze(1)  # [1,Q,T]
if seed_codes.ndim != 3 or seed_codes.shape[0] != 1:
    raise ValueError(f"❌ Unexpected shape for seed tokens: {seed_codes.shape}")

# Transformer expects [B,T,Q]
seed_tokens = seed_codes.squeeze(0).transpose(0, 1).unsqueeze(0).to(DEVICE)  # [1,T,Q]
seed_tokens = seed_tokens[:, :min(seed_tokens.shape[1], SEED_SECONDS * TOKENS_PER_SEC)]

# prepend BOS
bos = torch.full((1, 1, Q), BOS_ID, dtype=torch.long, device=DEVICE)
generated = torch.cat([bos, seed_tokens], dim=1)  # [1, 1+Tseed, Q]

print(f"🔁 Using warm-up seed from {first_token_path.name}, shape: {seed_tokens.shape}")

with torch.inference_mode():
    step_idx  = generated.shape[1]
    step_temp = TEMPERATURE

    while step_idx < TOKENS_TO_GEN:
        # 1) get logits for last `seq_len` context
        context     = generated[:, -seq_len:]
        logits_list = model(context)  # list of Q tensors [1, Tctx, V]

        # 2) collapse check every WINDOW frames
        if step_idx % WINDOW == 0:
            entropies  = [window_entropy(generated[0, :, q]) for q in range(Q)]
            n_collapse = sum(float(H < ENTROPY_TRIGGER) for H in entropies)
            step_temp = TEMPERATURE * (TEMP_BOOST if n_collapse >= 0.7 * Q else 1.0)

        # 3) slice out last-token logits
        last_logits = [lg[:, -1] for lg in logits_list]  # each [1,V]

        # 4) sample each head
        next_tokens = []
        for q, logit_q in enumerate(last_logits):
            # repetition penalty (exclude BOS/PAD)
            logit_q = apply_scaled_repetition_penalty(logit_q, generated[0, :, q], REPETITION_PENALTY)

            # temperature
            logit_q = logit_q / step_temp

            # hard masks
            logit_q[:, BOS_ID] = -float("inf")
            logit_q[:, PAD_ID] = -float("inf")

            # optional top-k
            if TOP_K and TOP_K > 0:
                topv, topi = torch.topk(logit_q, TOP_K, dim=-1)
                filt = torch.full_like(logit_q, -float("inf"))
                filt.scatter_(1, topi, topv)
                logit_q = filt

            # top-p or plain multinomial
            if TOP_P and TOP_P > 0:
                # nucleus sampling
                probs = torch.softmax(logit_q, dim=-1)
                probs, idx = torch.sort(probs, dim=-1, descending=True)
                cdf = torch.cumsum(probs, dim=-1)
                mask = cdf > TOP_P
                mask[..., 0] = False
                probs[mask] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_tok = idx.gather(-1, torch.multinomial(probs, 1))
            else:
                probs    = torch.softmax(logit_q, dim=-1)
                next_tok = torch.multinomial(probs, 1)

            next_tokens.append(next_tok.squeeze(-1))  # [1]

        # append new frame [1,1,Q]
        new_frame = torch.stack(next_tokens, dim=-1).unsqueeze(1)
        generated = torch.cat([generated, new_frame], dim=1)
        step_idx += 1

flat = generated.squeeze(0)              # [T, Q]
print("unique token IDs (full clip):", torch.unique(flat))
print(f"📝 Generated token sequence shape: {generated.shape}")

# strip BOS and transpose back to [1,Q,T]
tokens = generated.squeeze(0)                           # [T,Q]
tokens_no_bos = tokens[1:].T.unsqueeze(0).contiguous()  # [1,Q,T]

# sanity check
uniq_decoding = torch.unique(tokens_no_bos)
print("🔍 Unique IDs for decoding (after BOS strip):", uniq_decoding)
if (uniq_decoding >= 1024).any():
    print("⚠️ Warning: Some invalid tokens (≥1024) still present!")
else:
    print("✅ All decoding tokens within expected range (0-1023)")

TOK_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(tokens_no_bos.cpu(), TOK_OUT_PATH)
print(f"💾 Saved tokens → {TOK_OUT_PATH}")

# ──────────────────── DECODE TO AUDIO ────────────────────
codec = EncodecModel.encodec_model_48khz().to(DEVICE).eval()

# choose or construct a scale to match T, RETURN SHAPE [1,1,T]
def prepare_scale(target_len: int, Q: int, seed_scale):
    """Return scale tensor shaped [1,1,T] as required by Encodec.
    If we find a Q-channel scale, collapse to 1 channel by mean.
    """
    sc = None
    # 1) prefer seed scale
    if isinstance(seed_scale, torch.Tensor):
        sc = seed_scale
    else:
        # 2) try to grab a scale from any training file
        for p in TOKEN_DIR.rglob("*.th"):
            o = torch.load(p, map_location="cpu")
            if isinstance(o, dict) and isinstance(o.get("scale"), torch.Tensor):
                sc = o["scale"]
                break

    if sc is None:
        # 3) ones fallback
        return torch.ones(1, 1, target_len, device=DEVICE)

    # Normalize to [1, C, Tsrc] with C in {1, Q}
    if sc.ndim == 4 and sc.shape[1] == 1:
        sc = sc.squeeze(1)          # [1,1,T] or [1,Q,T]
    if sc.ndim != 3:
        sc = torch.ones(1, 1, target_len)

    # If scale has Q channels, collapse to 1 (mean avoids exploding memory)
    if sc.shape[1] == Q:
        sc = sc.mean(dim=1, keepdim=True)  # [1,1,T]
    elif sc.shape[1] != 1:
        sc = sc[:, :1, :]                  # conservative fallback to 1 ch

    # Fit length to target_len
    if sc.shape[-1] >= target_len:
        sc = sc[..., :target_len]
    else:
        reps = (target_len + sc.shape[-1] - 1) // sc.shape[-1]
        sc = sc.repeat(1, 1, reps)[..., :target_len]

    return sc.to(DEVICE).contiguous()

# Chunked decode to avoid OOM

def safe_decode(codec, tokens, scale, chunk_size=CHUNK_FRAMES):
    """Decode Encodec tokens in smaller chunks to avoid OOM.
    tokens: [1, Q, T]
    scale:  [1, 1, T]
    Returns a tensor [1, C, S] on CPU.
    """
    B, Qc, T = tokens.shape
    assert B == 1
    out_chunks = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        tok_chunk = tokens[:, :, start:end]
        sc_chunk  = scale[:, :, start:end]
        try:
            dec = codec.decode([(tok_chunk.to(DEVICE), sc_chunk.to(DEVICE))])[0]  # [B,C,S]
            dec = dec[0].cpu()  # [C,S]
        except torch.cuda.OutOfMemoryError:
            print("⚠️ CUDA OOM on chunk, retrying CPU decode …")
            torch.cuda.empty_cache()
            codec_cpu = codec.to("cpu")
            dec = codec_cpu.decode([(tok_chunk.cpu(), sc_chunk.cpu())])[0][0].cpu()
            codec.to(DEVICE)
        out_chunks.append(dec)

    return torch.cat(out_chunks, dim=-1).unsqueeze(0)  # [1,C,S_total]

Tneed = tokens_no_bos.shape[-1]
scale_use = prepare_scale(Tneed, Q, seed_scale)  # <- [1,1,T]

with torch.inference_mode():
    decoded = safe_decode(codec, tokens_no_bos, scale_use, chunk_size=CHUNK_FRAMES)

wav = decoded[0].squeeze().detach().cpu().numpy().astype("float32")
# normalize to prevent clipping
wav = (wav / max(1e-6, abs(wav).max())).clip(-1 + 1e-6, 1 - 1e-6)
wav = wav.T  # (T,2)

WAV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(WAV_OUT_PATH.as_posix(), wav, 48000, subtype="PCM_16")
print(f"🎧 Wrote {DESIRED_SECONDS}s clip to → {WAV_OUT_PATH}")
