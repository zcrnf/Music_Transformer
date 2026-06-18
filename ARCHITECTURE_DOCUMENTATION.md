# Music Transformer Architecture Documentation

## Overview
This system trains a GPT-2 style transformer to generate classical piano music, conditioned on composer style. It uses MIDI files as input and learns to predict the next musical event.

---

## 1. MIDI Files - What They Are

**MIDI (Musical Instrument Digital Interface)** is NOT audio - it's a sequence of musical instructions:

```
Time 0.0s: Note On, Pitch=60 (Middle C), Velocity=80
Time 0.5s: Note Off, Pitch=60
Time 0.5s: Note On, Pitch=64 (E), Velocity=75
Time 1.0s: Note Off, Pitch=64
```

**Why MIDI for AI?**
- Compact representation (10KB vs 3MB for audio)
- Discrete events (easier to model than continuous audio)
- Contains musical structure (pitch, rhythm, dynamics)
- Can be edited and regenerated perfectly

**Our Dataset:**
- 10,841 classical piano MIDI files
- 2,567 unique composers (Beethoven, Chopin, Bach, etc.)
- Organized by composer name for style conditioning

---

## 2. Tokenization - Converting MIDI to Numbers

### What is Tokenization?
Transformers can only process numbers, not MIDI events. Tokenization converts musical events into discrete tokens (integers).

### REMI Tokenization (REpresentation of MIdi)

**REMI breaks down music into 5 types of tokens:**

1. **Bar** - Marks the start of a measure
2. **Position** - Time within the bar (e.g., beat 1, beat 2)
3. **Pitch** - What note to play (21-108 = piano range A0 to C8)
4. **Velocity** - How hard to hit the key (32 bins for dynamics)
5. **Duration** - How long to hold the note

**Example tokenization:**
```
MIDI Event: Play Middle C (pitch 60) at beat 1 with medium loudness for 1 beat

Becomes 4 tokens:
[Position_1, Pitch_60, Velocity_16, Duration_8]
```

**Our Configuration:**
- Vocabulary size: 419 tokens total
- Beat resolution: 8 ticks/beat (fine timing)
- Special tokens: BOS (Begin), PAD (padding)
- ~5 tokens per note on average

**Variable Length Storage:**
- Pieces range from 4 tokens (very short) to 418,481 tokens (very long)
- Median: 7,780 tokens (~3 minutes of music)
- Mean: 13,086 tokens
- Training crops random 8192-token windows from each piece

---

## 3. Transformer Architecture

### High-Level Architecture

```
Input: [BOS, tok1, tok2, ..., tok8191]
   ↓
Token Embedding (419 vocab → 512 dimensions)
   ↓
Composer Embedding (2567 composers → 512 dimensions) [ADDED TO TOKENS]
   ↓
Dropout (10%)
   ↓
GPT-2 Transformer (8 layers)
   ↓
Output Head (512 → 419 vocab)
   ↓
Predict: [tok2, tok3, ..., tok8192]
```

### Detailed Component Breakdown

#### **A. Token Embedding**
- Maps each token ID (0-418) to a 512-dimensional vector
- Learned during training
- Captures musical meaning (e.g., similar pitches have similar vectors)

#### **B. Composer Conditioning**
- **Why?** Different composers have different styles (Beethoven ≠ Debussy)
- Each composer gets a learned 512-dimensional embedding
- Added to every token in the sequence
- Special ID 0 = "unconditional" (no style bias)
- Enables **Classifier-Free Guidance** during generation

#### **C. GPT-2 Transformer Blocks (8 layers)**

Each layer contains:

1. **Multi-Head Self-Attention (8 heads)**
   - Learns relationships between musical events
   - Example: "If there's a C major chord, the next note is likely C, E, or G"
   - 8 heads = 8 different "attention patterns" learned in parallel
   - **Causal masking**: Can only look at past tokens (autoregressive)

2. **Feed-Forward Network**
   - 2-layer MLP: 512 → 2048 → 512
   - Adds non-linearity and processing power
   - GELU activation (smooth, works better than ReLU)

3. **Layer Normalization + Residual Connections**
   - Stabilizes training
   - Helps gradients flow through 8 layers

**Total parameters:** ~50 million

#### **D. Output Head**
- Maps 512-dimensional hidden state → 419 logits (one per token)
- **Weight tying**: Shares weights with token embedding (reduces parameters)
- Softmax converts logits to probabilities
- Training: minimize cross-entropy loss (predict next token correctly)

---

## 4. Training Process

### Data Flow

```
1. Load MIDI file: Bach_Prelude.mid
2. Load tokens: [23, 145, 67, 234, ...] (12,000 tokens)
3. Random crop: Select tokens 3000-11191 (8192 tokens)
4. Prepend BOS: [415, tok3000, tok3001, ..., tok11191] (8193 tokens)
5. Split: input=[0:8192], target=[1:8193]
6. Batch: Combine 2 pieces
7. Forward pass: Predict next token for each position
8. Loss: Compare predictions to true next tokens
9. Backprop: Update weights to improve predictions
10. Repeat for 500 epochs
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Sequence Length** | 8192 tokens | Context window (long-range coherence) |
| **Batch Size** | 2 | Pieces processed simultaneously |
| **Gradient Accumulation** | 8 steps | Effective batch = 2×8 = 16 |
| **Learning Rate** | 5e-4 | Step size for weight updates |
| **Optimizer** | AdamW | Adaptive learning + weight decay |
| **Warmup** | 5% of steps | Gradual LR increase at start |
| **Scheduler** | Cosine decay | LR decreases over training |
| **Epochs** | 500 | Full passes through dataset |
| **Gradient Clipping** | 1.0 | Prevents exploding gradients |

### Training Loop (per epoch)

```python
for batch in dataloader:
    1. Load input_ids [B, 8192] and labels [B, 8192]
    2. Forward pass: logits = model(input_ids, composer_id)
    3. Compute loss: CrossEntropy(logits, labels)
    4. Backward pass: loss.backward()
    5. Gradient accumulation (every 8 batches):
       - Clip gradients to max norm 1.0
       - Update weights: optimizer.step()
       - Update learning rate: scheduler.step()
    6. Log: loss, grad norm, learning rate
```

### Optimizations

- **Mixed Precision (AMP)**: Uses FP16 for speed, FP32 for stability
- **Gradient Checkpointing**: Saves memory by recomputing activations
- **Weight Tying**: Token embedding = output head weights
- **Torch Compile**: JIT compilation for faster execution
- **PAD Masking**: Ignores padding tokens in loss calculation
- **Label Smoothing (0.1)**: Prevents overconfidence

---

## 5. Generation Process

### Autoregressive Sampling

```
1. Start with seed: [BOS, tok1, tok2, ..., tok127] (128 tokens)
2. Pass through model: logits = model(seed, composer_id=Chopin)
3. Extract last position logits: logits[-1] (419 values)
4. Apply temperature: logits / 0.9 (higher = more random)
5. Apply top-k (40): Keep only top 40 most likely tokens
6. Apply top-p (0.95): Keep tokens until cumulative prob > 0.95
7. Sample: next_token ~ Categorical(softmax(filtered_logits))
8. Append: seed = [seed, next_token]
9. Repeat steps 2-8 until 2048 tokens generated
10. Detokenize: Convert tokens → MIDI file
```

### Classifier-Free Guidance (CFG)

Enhances composer style adherence:

```python
# Generate with and without composer conditioning
logits_cond = model(input, composer_id=Chopin)
logits_uncond = model(input, composer_id=0)  # Unconditional

# Amplify the difference (scale=3.0)
logits_final = logits_uncond + 3.0 * (logits_cond - logits_uncond)

# Sample from enhanced distribution
next_token = sample(logits_final)
```

Higher scale → stronger style adherence

---

## 6. Key Design Decisions

### Why 8192 Context Length?
- Covers 95%+ of pieces in dataset
- Enough for long-range musical coherence (verse-chorus structure)
- Still fits in 144GB VRAM with room to spare

### Why REMI Tokenization?
- Preserves exact timing information
- Compact vocabulary (419 vs 20,000+ for MIDI-like)
- Better than raw MIDI events (more structured)
- Better than piano roll (maintains note timing)

### Why Composer Conditioning?
- Enables style transfer (generate "Bach in the style of Chopin")
- Improves sample quality (focused distributions)
- Allows control during generation

### Why GPT-2 Architecture?
- Proven for sequence modeling
- Causal attention matches music generation (left-to-right)
- Efficient implementation (HuggingFace Transformers)
- Scales well (can increase layers/dimensions)

---

## 7. File Structure

```
Transformer_ECS111/
├── data_raw/
│   ├── midis/                      # Original 10,841 MIDI files
│   └── midis_organized/            # Organized by composer
│       ├── Bach/
│       ├── Chopin/
│       └── ...
├── encoded_tokens/
│   └── midis/                      # Tokenized files (.th)
│       ├── Bach/
│       │   ├── piece1.th           # torch.Tensor([tok1, tok2, ...])
│       │   └── piece2.th
│       └── tokenizer.json          # REMI config
├── model_results_midis/
│   ├── music_transformer_ep10.pt   # Checkpoint at epoch 10
│   ├── music_transformer_ep20.pt   # Checkpoint at epoch 20
│   └── ...
├── scripts/
│   ├── 0_midi_organization.py      # Organize by composer
│   ├── 1_midi_tokenization.py      # MIDI → tokens
│   ├── 3_midi_data_preparation.py  # Generate metadata
│   ├── 4_Transformer.py            # Training script
│   └── 5_MusicPrediction.py        # Generation script
├── metadata_clean_midis.jsonl      # token_path, composer_id, seq_len
├── composer_mapping.json           # composer_name → composer_id
└── training.log                    # Training progress
```

---

## 8. Mathematical Formulation

### Attention Mechanism

```
Q = Input × W_Q    (Query)
K = Input × W_K    (Key)
V = Input × W_V    (Value)

Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Causal Mask:**
```
Mask[i, j] = 0 if j > i else 1
# Position i can only attend to positions ≤ i
```

### Loss Function

```
L = -Σ log P(token_t | tokens_<t, composer_id)

Where:
P(token_t | ...) = softmax(model_output)[token_t]
```

Ignores PAD tokens: `loss_mask = (labels != PAD_ID)`

---

## 9. Performance Metrics

### Training Speed
- **~4.5 iterations/second** (5421 batches/epoch)
- **~20 minutes per epoch**
- **~167 hours for 500 epochs**

### Model Capacity
- **Parameters:** ~50M
- **Context Window:** 8192 tokens (~10-15 minutes of music)
- **Memory Usage:** ~3GB VRAM during training

### Dataset Statistics
- **Files:** 10,841 MIDI pieces
- **Composers:** 2,567 unique
- **Tokens:** ~142 million total
- **Top Composers:** Scarlatti (279), Bach (246), Liszt (197)

---

## 10. Future Improvements

1. **Increase model size**: 8 layers → 12-16 layers
2. **Flash Attention**: 2-3x faster self-attention
3. **Rotary Position Embeddings**: Better long-range modeling
4. **Hierarchical modeling**: Separate models for structure + details
5. **Multi-instrument**: Extend beyond piano
6. **Rhythm-aware tokens**: Better timing representation
7. **Data augmentation**: Transpose, time-stretch MIDI

---

## Summary

This system learns the statistical patterns of classical piano music by:
1. Converting MIDI files into discrete tokens (REMI)
2. Training a transformer to predict the next token
3. Conditioning on composer style for controllable generation
4. Using autoregressive sampling to generate new music

The result: A model that can generate coherent, stylistically-appropriate piano music in the style of any composer in the dataset.
