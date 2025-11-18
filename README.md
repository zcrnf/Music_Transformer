# Music Transformer - REMI Tokenization Training Pipeline

A GPT-style transformer trained on MIDI music files using REMI tokenization with composer conditioning. 

Sample result: see the midi file in the folder

Update: For people who have visited this project before: this project switched the training data from real piano to midi, switched tokenization from Encodec to REMI, for a better performance

Training Data Source: https://drive.google.com/drive/folders/1gskpFXA04GMyTT0f9r4z3C2ddzCEOP42

## Overview

This project trains a music generation model on ~10,841 MIDI files from 237 composers using:
- **REMI tokenization** (vocab size: 419 tokens)
- **GPT2-like architecture** (8 layers, 8 heads, d_model=512, context=8192)
- **Composer conditioning** with classifier-free guidance (CFG)
- **Mixed precision training** (AMP + TF32 on H200 NVL)

## Quick Start

### 1. Environment Setup

```bash
cd /root/Transformer_ECS111

# Install dependencies (if needed)
pip install torch miditok symusic pretty_midi tqdm
```

### 2. Data Preparation (One-time Setup)

#### Step 1: Organize MIDI Files by Composer
```bash
python scripts/0_midi_organization.py
```
- **Input**: `data_raw/midis/*.mid` (raw MIDI files)
- **Output**: `data_raw/midis_organized/<Composer>/<file>.mid`

#### Step 2: Tokenize MIDI Files
```bash
python scripts/1_midi_tokenization.py
```
- **Input**: `data_raw/midis_organized/`
- **Output**: `encoded_tokens/midis/*.th` (tokenized sequences)
- **Tokenizer**: `encoded_tokens/midis/tokenizer.json`
- **Vocab**: 419 tokens (PAD=0, BOS=1, EOS=2, plus REMI events)

#### Step 3: Generate Metadata with Composer Bucketing
```bash
python scripts/3_midi_data_preparation.py
```
- **Input**: `encoded_tokens/midis/*.th`
- **Output**: 
  - `metadata_clean_midis.jsonl` (file paths + composer IDs)
  - `composer_mapping.json` (237 frequent composers + OTHER bucket)
- **Bucketing**: Composers with ≥10 pieces get unique IDs (1-237), rare composers → ID 238

### 3. Training

```bash
cd /root/Transformer_ECS111
python scripts/4_Transformer.py
```

**Training Configuration**:
- **Batch size**: 2 (effective 16 with gradient accumulation 8)
- **Sequence length**: 8192 tokens
- **Optimizer**: AdamW (LR=5e-4, cosine schedule, 5% warmup)
- **Epochs**: 500 (saves checkpoint every epoch)
- **Checkpoints**: 
  - Linux: `model_results_midis/music_transformer_ep{N}.pt`
  - Windows WSL: `/mnt/c/Users/zhengmy/Transformer_ECS111/music_transformer_ep{N}.pt`
- **Log**: `training.log` (epoch-level summaries)

**Training Progress** (from test run):
- Epoch 1: loss = 8.55 (~18 min)
- Epoch 2: loss = 3.98 (~18 min)
- Speed: ~5 iterations/sec on H200 NVL

**Stop Training**: Press `Ctrl+C` to stop gracefully

### 4. Music Generation

```bash
cd /root/Transformer_ECS111
python scripts/5_MusicPrediction.py
```

**Interactive Prompts**:
1. **Epoch number**: Which checkpoint to load (e.g., `2`)
2. **Temperature**: Sampling randomness (0.7-1.2, default: `0.9`)
3. **Top-k**: Keep top-k tokens (20-50, default: `40`, 0=disable)
4. **Top-p**: Nucleus sampling threshold (0.9-0.98, default: `0.95`)
5. **Repetition penalty**: Discourage repeats (1.0-1.1, default: `1.02`)
6. **Seed length**: Tokens to prime generation (64-256, default: `128`)
7. **Composer ID**: 
   - `0` = Unconditional (no style)
   - `1-237` = Specific composer (see `composer_mapping.json`)
   - `238` = OTHER (rare composers)
8. **Guidance scale**: CFG strength (1.0=off, 3.0=moderate, 7.0=strong)

**Output**:
- **MIDI file**: `generated_music/ep{N}_composer{ID}.mid`
- **Raw tokens**: `generated_music/ep{N}_composer{ID}.th` (fallback)

**Example Generation Session**:
```
Enter the epoch number to load: 2
Enter temperature (e.g. 0.9): 0.9
Enter top-k (e.g. 40, 0 to disable): 40
Enter top-p (e.g. 0.95): 0.95
Enter repetition penalty (e.g. 1.02): 1.02
Enter seed length in tokens (e.g. 128): 128
Enter composer ID (0 for unconditional): 0
Enter guidance scale (1.0=no guidance, 3.0=moderate, 7.0=strong): 1.0
```

Result: `generated_music/ep2_composer0.mid` (2048 tokens, ~40 seconds)

## File Structure

```
/root/Transformer_ECS111/
├── scripts/
│   ├── 0_midi_organization.py          # Organize MIDIs by composer
│   ├── 1_midi_tokenization.py          # REMI tokenization
│   ├── 3_midi_data_preparation.py      # Metadata + composer bucketing
│   ├── 4_Transformer.py                # Training script
│   └── 5_MusicPrediction.py            # Generation script
├── data_raw/
│   ├── midis/                          # Original MIDI files
│   └── midis_organized/                # Organized by composer
├── encoded_tokens/
│   └── midis/                          # Tokenized .th files
│       └── tokenizer.json              # REMI tokenizer config
├── model_results_midis/                # Checkpoints (Linux)
│   └── music_transformer_ep*.pt
├── generated_music/                    # Generated MIDI outputs
│   └── ep*_composer*.mid
├── metadata_clean_midis.jsonl          # Training metadata
├── composer_mapping.json               # Composer ID mapping
└── training.log                        # Training logs
```

## Model Architecture

### Transformer Configuration
```python
- Layers: 8
- Attention heads: 8
- Embedding dimension: 512
- Feed-forward dimension: 2048
- Context length: 8192 tokens
- Dropout: 0.1
- Parameters: ~25-30M
```

### Special Features
1. **Rotary Position Embeddings (RoPE)**: Better long-range modeling
2. **Gradient Checkpointing**: Memory efficiency for long sequences
3. **Weight Tying**: Token embedding ↔ output projection
4. **Logit Scaling**: `output * (d_model ** -0.5)` for stability
5. **Composer Conditioning**: Scaled embedding (scale=0.25)
6. **Classifier-Free Guidance (CFG)**: 60% unconditional training dropout

### Training Optimizations
- **Mixed Precision (AMP)**: FP16 gradients, FP32 optimizer states
- **TF32**: Faster matmul on Ampere GPUs
- **Gradient Clipping**: Max norm 1.0
- **Label Smoothing**: 0.1 for regularization
- **Token Dropout**: 1% to PAD for denoising

## Composer IDs

### Frequent Composers (≥10 pieces each)
See `composer_mapping.json` for full list. Examples:
- `11`: Bach
- `20`: Beethoven
- `31`: Brahms
- `44`: Chopin
- `50`: Debussy
- `122`: Liszt
- `148`: Mozart
- `171`: Rachmaninoff
- `173`: Ravel
- `193`: Schubert
- `195`: Schumann
- `219`: Tchaikovsky

### Special IDs
- `0`: Unconditional (no composer style)
- `238`: OTHER (rare composers with <10 pieces)

## Troubleshooting

### Training Issues

**Problem**: High initial loss (>100)
- **Cause**: Vocab mismatch or incorrect special token IDs
- **Fix**: Ensure `VOCAB_SIZE=419`, `PAD_ID=0`, `BOS_ID=1`, `EOS_ID=2`

**Problem**: Loss explosion or NaN
- **Cause**: Gradient overflow or learning rate too high
- **Fix**: Check gradient clipping, reduce LR, verify logit scaling

**Problem**: Slow training speed (<3 it/s)
- **Cause**: Large batch size with long sequences (O(L²) attention)
- **Fix**: Use batch_size=2, check gradient accumulation instead

### Generation Issues

**Problem**: MIDI file won't play
- **Possible causes**:
  1. Invalid token sequences (check for out-of-range tokens)
  2. Miditok decoding errors (check library version)
  3. Empty or corrupted MIDI file (verify file size >0)
- **Debug**: Check raw tokens in `.th` file, try different seed/parameters

**Problem**: Repetitive or nonsensical music
- **Cause**: Insufficient training or poor sampling parameters
- **Fix**: 
  - Train for more epochs (loss should be <2.0)
  - Increase temperature (0.9-1.1)
  - Lower top-p (0.90-0.95)
  - Increase repetition penalty (1.05-1.1)

**Problem**: Generation crashes with CUDA OOM
- **Cause**: Generating too many tokens at once
- **Fix**: Reduce target tokens in script or use CPU for generation

## Performance Metrics

### Training (H200 NVL)
- **Speed**: 4.9-5.0 it/s
- **Memory**: ~40GB VRAM (with gradient checkpointing)
- **Time per epoch**: ~18 minutes (10,841 sequences)

### Generation
- **Speed**: ~10-20 tokens/sec (depends on temperature/sampling)
- **Memory**: ~5GB VRAM for 2048 token generation

## Advanced Usage

### Resume Training from Checkpoint
Edit `4_Transformer.py` line 393:
```python
train_loop(model, dl, opt, scheduler, scaler, DEVICE, ACCUM_STEPS, 
           MAX_NORM, START_EPOCH, EPOCHS, OUTPUT_DIR, meta, log_file)
```
Change `START_EPOCH = 0` to desired epoch (e.g., `START_EPOCH = 2`)

### Custom Sampling Parameters
Edit `5_MusicPrediction.py` defaults:
```python
TEMPERATURE = 0.9      # Higher = more random
TOP_K = 40            # Smaller = more focused
TOP_P = 0.95          # Lower = more deterministic
REP_PENALTY = 1.02    # Higher = less repetition
```

### Batch Generation
Modify `5_MusicPrediction.py` to loop over composer IDs:
```python
for composer_id in range(1, 238):
    # Set COMPOSER_ID = composer_id
    # Generate and save
```

## Citation

If you use this code, please cite:
- **REMI Tokenization**: Huang & Yang (2020)
- **Transformer Architecture**: Vaswani et al. (2017)
- **Classifier-Free Guidance**: Ho & Salimans (2022)

## License

This project uses the following libraries:
- PyTorch (BSD License)
- miditok (Apache 2.0)
- symusic (MIT License)

## Contact

For questions or issues, check:
1. Training logs: `training.log`
2. Error messages in terminal output
3. Generated raw tokens: `generated_music/*.th`
