#!/usr/bin/env python3
"""
MIDI Tokenization Script - VARIABLE LENGTH
Stores full MIDI pieces without truncation/padding.
Training will handle random cropping (like audio pipeline does).
"""

import sys
import os
from pathlib import Path
import json
import torch
from miditok import REMI, TokenizerConfig
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

MIDI_DIR = Path("data_raw/midis_organized")
OUTPUT_DIR = Path("encoded_tokens/midis")

# NO MAX_SEQ_LEN - store variable length!
# Training will crop to desired context window (1024, 2048, 4096, etc.)

# Tokenizer configuration optimized for piano music
# REMI tokenization: ~5 tokens per note
# (Pitch, Velocity, Duration, Position, Bar/Beat markers)
TOKENIZER_PARAMS = {
    "pitch_range": (21, 108),        # Piano range (A0 to C8)
    "beat_res": {(0, 4): 8, (4, 12): 4},  # Fine timing: 8 ticks/beat (0-4 bars), 4 ticks/beat after
    "num_velocities": 32,             # Velocity bins (MIDI 0-127 → 32 bins)
    "special_tokens": ["PAD", "BOS", "EOS"],  # Control tokens
    "use_chords": False,              # Disable chord detection (simpler vocab)
    "use_rests": True,                # Include rest events (silence)
    "use_tempos": True,               # Include tempo changes
    "use_time_signatures": True,      # Include time signature changes
    "use_programs": False,            # Single instrument (piano only)
}

# ═══════════════════════════════════════════════════════════
# TOKENIZER SETUP
# ═══════════════════════════════════════════════════════════

def create_tokenizer():
    """Initialize REMI tokenizer with piano-optimized settings."""
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)
    
    # Save tokenizer configuration
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))
    
    return tokenizer

# ═══════════════════════════════════════════════════════════
# TOKENIZATION
# ═══════════════════════════════════════════════════════════

def tokenize_midi_file(midi_path: Path, tokenizer: REMI) -> torch.Tensor:
    """
    Tokenize a single MIDI file - VARIABLE LENGTH (no padding/truncation).
    
    Args:
        midi_path: Path to MIDI file
        tokenizer: REMI tokenizer instance
    
    Returns:
        token_tensor: 1D tensor of shape [T] where T = actual token count
    """
    # Tokenize MIDI
    tokens = tokenizer(midi_path)
    
    # Convert to tensor (tokens is a TokSequence object with .ids attribute)
    if hasattr(tokens, 'ids'):
        token_ids = tokens.ids
    else:
        token_ids = tokens
    
    # Handle multi-track MIDI (take first track if multiple)
    if isinstance(token_ids, list) and len(token_ids) > 0:
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]  # Take first track
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
    else:
        token_tensor = torch.as_tensor(token_ids, dtype=torch.long)
    
    # Ensure 1D
    if token_tensor.ndim > 1:
        token_tensor = token_tensor.flatten()
    
    # NO truncation or padding - store full length!
    return token_tensor

# ═══════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════

def tokenize_dataset():
    """Process all MIDI files and save tokenized versions."""
    
    # Initialize tokenizer
    print("🎹 Initializing REMI tokenizer...", flush=True)
    tokenizer = create_tokenizer()
    
    # Get all MIDI files
    midi_files = sorted([f for f in MIDI_DIR.rglob("*.mid") if f.is_file()])
    midi_files.extend(sorted([f for f in MIDI_DIR.rglob("*.midi") if f.is_file()]))
    
    if not midi_files:
        print(f"❌ No MIDI files found in {MIDI_DIR.resolve()}", flush=True)
        return
    
    print(flush=True)
    print("="*60, flush=True)
    print("🚀 MIDI TOKENIZATION (VARIABLE LENGTH)", flush=True)
    print("="*60, flush=True)
    print(f"📂 Input directory:  {MIDI_DIR.resolve()}", flush=True)
    print(f"📂 Output directory: {OUTPUT_DIR.resolve()}", flush=True)
    print(f"📝 Vocabulary size:  {len(tokenizer)} tokens", flush=True)
    print(f"🎵 Total MIDI files: {len(midi_files)}", flush=True)
    print(f"💾 Storage mode:     VARIABLE LENGTH (no padding/truncation)", flush=True)
    print("="*60, flush=True)
    print(flush=True)
    
    success_count = 0
    error_count = 0
    total_tokens = 0
    token_lengths = []
    
    # Process each MIDI file with progress bar
    # Force tqdm to write to stderr and flush immediately
    for midi_path in tqdm(midi_files, desc="🎼 Tokenizing", unit="file", 
                          ncols=80, file=sys.stderr, mininterval=0.1):
        try:
            # Create output path preserving directory structure
            relative_path = midi_path.relative_to(MIDI_DIR)
            token_path = OUTPUT_DIR / relative_path.with_suffix(".th")
            
            # Skip if already tokenized
            if token_path.exists():
                success_count += 1
                continue
            
            # Tokenize MIDI file (variable length)
            tokens = tokenize_midi_file(midi_path, tokenizer)
            
            token_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save tokens (variable length!)
            torch.save(tokens, token_path)
            
            # Update statistics
            success_count += 1
            token_len = tokens.shape[0]
            total_tokens += token_len
            token_lengths.append(token_len)
            
            # Print every 100 files
            if success_count % 100 == 0:
                avg_len = sum(token_lengths[-100:]) / min(100, len(token_lengths))
                print(f"\n✓ {success_count}/{len(midi_files)} files | Avg length (last 100): {avg_len:.0f} tokens", 
                      file=sys.stderr, flush=True)
            
        except Exception as e:
            error_count += 1
            tqdm.write(f"❌ {midi_path.name}: {e}", file=sys.stderr)
    
    # Calculate statistics
    token_lengths.sort()
    min_len = min(token_lengths) if token_lengths else 0
    max_len = max(token_lengths) if token_lengths else 0
    avg_len = total_tokens / success_count if success_count > 0 else 0
    median_len = token_lengths[len(token_lengths)//2] if token_lengths else 0
    
    # Print final statistics
    print(flush=True)
    print("="*60, flush=True)
    print("🎉 TOKENIZATION COMPLETE!", flush=True)
    print("="*60, flush=True)
    print(f"✅ Successfully processed: {success_count}/{len(midi_files)} files", flush=True)
    print(f"❌ Errors: {error_count}", flush=True)
    print(flush=True)
    print("📊 TOKEN LENGTH STATISTICS:", flush=True)
    print(f"   Total tokens:        {total_tokens:,}", flush=True)
    print(f"   Average length:      {avg_len:.0f} tokens", flush=True)
    print(f"   Median length:       {median_len:,} tokens", flush=True)
    print(f"   Min length:          {min_len:,} tokens", flush=True)
    print(f"   Max length:          {max_len:,} tokens", flush=True)
    print(flush=True)
    print("💡 TRAINING RECOMMENDATIONS:", flush=True)
    print(f"   For avg piece ({avg_len:.0f} tokens):", flush=True)
    print(f"   - Use seq_len=1024 (crop ~{100*1024/avg_len:.0f}% per sample)", flush=True)
    print(f"   - Use seq_len=2048 (crop ~{100*2048/avg_len:.0f}% per sample)", flush=True)
    print(f"   - Use seq_len=4096 (crop ~{100*4096/avg_len:.0f}% per sample)", flush=True)
    print(flush=True)
    print(f"📁 Output directory: {OUTPUT_DIR.resolve()}", flush=True)
    print(f"💾 Tokenizer config: {OUTPUT_DIR / 'tokenizer.json'}", flush=True)
    print(f"📝 Vocabulary size:  {len(tokenizer)} tokens", flush=True)
    print("="*60, flush=True)

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tokenize_dataset()
