#!/usr/bin/env python
"""
Generate MIDI music with the MIDITransformer trained on REMI tokens.
Features:
- Loads MIDITransformer checkpoint
- Uses Classifier-Free Guidance for composer conditioning
- Top-k/top-p sampling with repetition penalty
- Outputs .mid file using miditok detokenization
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from miditok import REMI, TokSequence
import json

# ───────────────────────── USER CONFIG ─────────────────────────
CHECKPOINT_EPOCH   = int(input("Enter the epoch number to load: "))
DESIRED_TOKENS     = 2048  # ~80-100 seconds of music at median density

TEMPERATURE        = float(input("Enter temperature (e.g. 0.9): ") or 0.9)
TOP_K              = int(input("Enter top-k (e.g. 40, 0 to disable): ") or 40)
TOP_P              = float(input("Enter top-p (e.g. 0.95): ") or 0.95)
REPETITION_PENALTY = float(input("Enter repetition penalty (e.g. 1.02): ") or 1.02)
SEED_TOKENS        = int(input("Enter seed length in tokens (e.g. 128): ") or 128)

# Classifier-Free Guidance
print("\n📌 Composer Selection:")
print("Loading composer mapping...")
with open("composer_mapping.json", "r") as f:
    COMPOSER_MAP = json.load(f)

# Show top composers
top_composers = sorted(COMPOSER_MAP.items(), key=lambda x: x[0])[:20]
print("\nTop 20 composers (alphabetically):")
for i, (name, cid) in enumerate(top_composers):
    print(f"  {cid}: {name}")
print(f"  0: Unconditional (no style conditioning)")

COMPOSER_ID        = int(input("\nEnter composer ID (0 for unconditional): ") or 0)
GUIDANCE_SCALE     = float(input("Enter guidance scale (1.0=no guidance, 3.0=moderate, 7.0=strong): ") or 3.0)

COMPOSER_SCALE = 0.25  # Scale factor for composer conditioning (gentle bias)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Special token IDs from tokenizer (miditok REMI)
PAD_ID     = 0   # PAD_None
BOS_ID     = 1   # BOS_None
EOS_ID     = 2   # EOS_None
VOCAB_SIZE = 419  # Full MIDI vocab size (0-418)

CHECKPOINT_PATH  = Path(f"model_results_midis/music_transformer_ep{CHECKPOINT_EPOCH}.pt")
MIDI_OUT_PATH    = Path(f"generated_music/ep{CHECKPOINT_EPOCH}_composer{COMPOSER_ID}.mid")
TOKEN_DIR        = Path("encoded_tokens/midis")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────── MODEL (matching training) ────────────────────
class MIDITransformer(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, n_layer, n_head, num_composers=2567):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_composers = num_composers

        # Single embedding layer for MIDI tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        
        # Composer conditioning embedding (index 0 reserved for unconditional/null)
        # No padding_idx here since 0 is a valid conditioning value
        self.composer_embedding = nn.Embedding(num_composers + 1, d_model)
        
        self.dropout = nn.Dropout(0.1)

        cfg = GPT2Config(
            vocab_size=1,
            n_positions=seq_len,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.transformer = GPT2Model(cfg)

        # Single output head with weight tying
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

    def _compute_attn_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != PAD_ID).long()

    def forward(self, input_ids: torch.Tensor, composer_id: torch.Tensor = None, labels: torch.Tensor = None):
        B, T = input_ids.shape

        # Default to unconditional if not provided
        if composer_id is None:
            composer_id = torch.zeros(B, dtype=torch.long, device=input_ids.device)

        # Token embeddings + composer conditioning (scaled to act as gentle bias)
        x = self.token_embedding(input_ids)  # [B, T, d_model]
        comp_emb = self.composer_embedding(composer_id).unsqueeze(1)  # [B, 1, d_model]
        x = x + COMPOSER_SCALE * comp_emb  # Broadcast scaled composer embedding across time
        x = self.dropout(x)

        hidden = self.transformer(
            inputs_embeds=x,
            attention_mask=self._compute_attn_mask(input_ids)
        ).last_hidden_state

        logits = self.lm_head(hidden) * (self.d_model ** -0.5)  # Scale logits for stability

        if labels is None:
            return logits

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            label_smoothing=0.1,
        )
        return loss

# ──────────────────── LOAD CHECKPOINT ────────────────────
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
d_model    = ckpt['d_model']
seq_len    = ckpt['seq_len']
n_layer    = ckpt['n_layer']
n_head     = ckpt['n_head']
vocab_size = ckpt.get('vocab_size', VOCAB_SIZE)
num_composers = ckpt.get('num_composers', 2567)

model = MIDITransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    seq_len=seq_len,
    n_layer=n_layer,
    n_head=n_head,
    num_composers=num_composers
).to(DEVICE).eval()

state = strip_prefixes(ckpt['model_state_dict'])
model.load_state_dict(state, strict=True)
print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} (seq_len={seq_len}, d_model={d_model})")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ──────────────────── SAMPLING HELPERS ────────────────────
def apply_repetition_penalty(logits, generated_tokens, penalty=1.02):
    """Apply repetition penalty to previously generated tokens."""
    vocab_size = logits.size(-1)
    valid = (generated_tokens >= 0) & (generated_tokens < vocab_size)
    ids = generated_tokens[valid]
    if ids.numel() == 0:
        return logits
    counts = torch.bincount(ids, minlength=vocab_size).float()
    scaling = penalty ** counts
    logits = logits / scaling.unsqueeze(0)
    return torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

# ──────────────────── TOKEN GENERATION ────────────────────
# Load one seed clip from the dataset
try:
    # Try to find a file from the selected composer if not unconditional
    if COMPOSER_ID != 0:
        composer_name = [k for k, v in COMPOSER_MAP.items() if v == COMPOSER_ID]
        if composer_name:
            composer_files = list((TOKEN_DIR / composer_name[0]).rglob("*.th"))
            if composer_files:
                first_token_path = composer_files[0]
            else:
                first_token_path = next(TOKEN_DIR.rglob("*.th"))
        else:
            first_token_path = next(TOKEN_DIR.rglob("*.th"))
    else:
        first_token_path = next(TOKEN_DIR.rglob("*.th"))
except StopIteration:
    raise FileNotFoundError(f"No .th files under {TOKEN_DIR}")

seed_tokens = torch.load(first_token_path, map_location="cpu")  # [T]
if seed_tokens.ndim != 1:
    raise ValueError(f"Expected 1D token tensor, got shape {seed_tokens.shape}")

# Clamp seed tokens to valid range
seed_tokens = torch.clamp(seed_tokens, 0, VOCAB_SIZE - 1)

# Take first SEED_TOKENS as seed (ensure we have at least 1 token)
actual_seed_len = max(1, min(len(seed_tokens), SEED_TOKENS))
seed_tokens = seed_tokens[:actual_seed_len].unsqueeze(0).to(DEVICE)  # [1, T]

# Prepend BOS
bos = torch.tensor([[BOS_ID]], dtype=torch.long, device=DEVICE)
generated = torch.cat([bos, seed_tokens], dim=1)  # [1, 1+T]

print(f"🔁 Using seed from {first_token_path.name}, length: {seed_tokens.shape[1]} tokens")
print(f"🎼 Generating with composer_id={COMPOSER_ID}, guidance_scale={GUIDANCE_SCALE}")

# Prepare composer tensors for CFG
composer_tensor = torch.tensor([COMPOSER_ID], dtype=torch.long, device=DEVICE)
unconditional_tensor = torch.tensor([0], dtype=torch.long, device=DEVICE)

with torch.inference_mode():
    step_idx = generated.shape[1]
    consecutive_fails = 0
    max_consecutive_fails = 50  # Stop if we fail to generate 50 times in a row
    
    while step_idx < DESIRED_TOKENS:
        try:
            # Get logits for last seq_len context
            context = generated[:, -seq_len:]
            
            # Classifier-Free Guidance
            if GUIDANCE_SCALE != 1.0 and COMPOSER_ID != 0:
                logits_cond = model(context, composer_id=composer_tensor)      # [1, T, V]
                logits_uncond = model(context, composer_id=unconditional_tensor)
                
                # Apply CFG: uncond + scale * (cond - uncond)
                logits = logits_uncond + GUIDANCE_SCALE * (logits_cond - logits_uncond)
            else:
                logits = model(context, composer_id=composer_tensor)
            
            # Check for NaN/Inf logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"⚠️  Warning: NaN/Inf logits at step {step_idx}, skipping...")
                consecutive_fails += 1
                if consecutive_fails >= max_consecutive_fails:
                    print(f"❌ Too many consecutive failures, stopping generation")
                    break
                continue
            
            # Get last token logits
            last_logits = logits[:, -1]  # [1, V]
            
            # Apply repetition penalty
            last_logits = apply_repetition_penalty(last_logits, generated[0], REPETITION_PENALTY)
            
            # Temperature (clamp to avoid extreme values)
            temp = max(0.01, min(TEMPERATURE, 10.0))
            last_logits = last_logits / temp
            
            # Mask special tokens
            last_logits[:, BOS_ID] = -float("inf")
            last_logits[:, PAD_ID] = -float("inf")
            
            # Top-k filtering
            if TOP_K and TOP_K > 0:
                topv, topi = torch.topk(last_logits, min(TOP_K, last_logits.size(-1)), dim=-1)
                filt = torch.full_like(last_logits, -float("inf"))
                filt.scatter_(1, topi, topv)
                last_logits = filt
            
            # Top-p (nucleus) sampling
            if TOP_P and TOP_P > 0:
                probs = torch.softmax(last_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf > TOP_P
                mask[..., 0] = False  # Keep at least the top token
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            
            # Clamp generated token to valid range
            next_token = torch.clamp(next_token, 0, VOCAB_SIZE - 1)
            
            # Append new token
            generated = torch.cat([generated, next_token], dim=1)
            step_idx += 1
            consecutive_fails = 0  # Reset on success
            
            # Progress update
            if step_idx % 100 == 0:
                print(f"  Generated {step_idx}/{DESIRED_TOKENS} tokens...")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU OOM during generation, stopping early")
                break
            else:
                raise

print(f"✅ Generated {generated.shape[1]} tokens total")

# Remove BOS token
tokens = generated[0, 1:].cpu().tolist()  # Convert to list of integers

# ──────────────────── DECODE TO MIDI ────────────────────
# Load the tokenizer config
tokenizer = REMI(params="encoded_tokens/midis/tokenizer.json")

print(f"🎹 Detokenizing {len(tokens)} tokens to MIDI...")

try:
    # miditok expects 2D input: batch of sequences
    # Wrap our single sequence in a list to make it [[token1, token2, ...]]
    tokens_batch = [tokens]
    
    # Decode returns symusic Score object directly (not a list)
    midi = tokenizer.decode(tokens_batch)

    # Ensure output directory exists
    MIDI_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # symusic Score export
    midi.dump_midi(str(MIDI_OUT_PATH))

    print(f"🎵 Saved MIDI file to {MIDI_OUT_PATH}")
    print(f"   Duration: ~{len(tokens) // 50} seconds (estimated)")
    comp_name = 'Unconditional'
    if COMPOSER_ID != 0:
        # COMPOSER_MAP may contain nested structure; try composer_to_id reverse lookup if present
        if 'composer_to_id' in COMPOSER_MAP:
            rev = {v: k for k, v in COMPOSER_MAP['composer_to_id'].items()}
            comp_name = rev.get(COMPOSER_ID, f'ID_{COMPOSER_ID}')
        else:
            comp_name = [k for k, v in COMPOSER_MAP.items() if v == COMPOSER_ID][0]
    print(f"   Composer: {COMPOSER_ID} ({comp_name})")
except Exception as e:
    MIDI_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"❌ Error during MIDI detokenization: {e}")
    print("Saving raw tokens for debugging...")
    torch.save(torch.tensor(tokens), MIDI_OUT_PATH.with_suffix('.th'))
    print(f"💾 Saved raw tokens to {MIDI_OUT_PATH.with_suffix('.th')}")
