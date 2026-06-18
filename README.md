# Music Transformer - Chopin Style Music Generation

An end-to-end ML pipeline for generating classical piano music in the style of Chopin using Transformer models.

## 🎹 Project Overview

This project implements a complete data engineering and machine learning pipeline:

1. **Data Collection**: Web scraping of 200+ Chopin MIDI files
2. **Data Preprocessing**: MIDI validation, splitting, and cleaning
3. **Tokenization**: Converting MIDI to model-ready sequential format using REMI tokenizer
4. **Model Training**: 229M parameter GPT-2 style Transformer (18 layers, 1024 dim)
5. **Music Generation**: Generate novel Chopin-style piano compositions

## 📁 Project Structure

```
├── scripts/
│   ├── 1_Preprocessing.py      # MIDI file validation and preprocessing
│   ├── 2_Tokenization.py       # Convert MIDI to tokens
│   ├── 3_Transformer.py        # Training script (single/multi-GPU)
│   └── 4_MIDI_Inference.py     # Music generation
├── midworld_chopin_downloader.py  # Data collection
├── tokenizer_config.json       # REMI tokenizer configuration
├── metadata_midi.jsonl         # Dataset metadata
├── launch_distributed.sh       # Distributed training launcher
├── slurm_job.sh               # SLURM cluster job script
├── deepspeed_config.json      # DeepSpeed configuration
└── cost_calculator.py         # Training cost estimator
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch transformers miditok pretty_midi tqdm numpy
```

### 2. Download Data

```bash
python midworld_chopin_downloader.py
```

### 3. Preprocess & Tokenize

```bash
python scripts/1_Preprocessing.py
python scripts/2_Tokenization.py
```

### 4. Train Model

**Single GPU:**
```bash
python scripts/3_Transformer.py
```

**Multi-GPU (Distributed):**
```bash
python scripts/3_Transformer.py --distributed
```

**Specific number of GPUs:**
```bash
python scripts/3_Transformer.py --distributed --world-size 4
```

### 5. Generate Music

```bash
python scripts/4_MIDI_Inference.py
```

## 🔧 Model Architecture

- **Type**: GPT-2 based causal language model
- **Parameters**: 229M
- **Architecture**: 18 transformer layers, 1024 embedding dimension, 8 attention heads
- **Sequence Length**: 1536 tokens
- **Training**: 50 epochs, AdamW optimizer, cosine learning rate schedule

## 🎯 Features

### Distributed Training
- **PyTorch DDP**: Native multi-GPU support
- Automatic scaling from 1 to N GPUs
- Checkpoint saving and resumption
- Mixed precision training (FP16)

### Data Pipeline
- MIDI file validation (tempo, duration, instruments)
- Automatic train/validation splitting
- REMI tokenization with configurable parameters
- Handles multitrack MIDI files

### Cost-Optimized Training
- Spot instance support
- Gradient accumulation
- Mixed precision training
- Efficient data loading

## 📊 Training Performance

| Setup | Time (50 epochs) | Cost (Spot) |
|-------|-----------------|-------------|
| 1 GPU | ~21 min | $0.57 |
| 4 GPUs | ~7 min | $0.73 |

See [COST_ANALYSIS.md](COST_ANALYSIS.md) for detailed estimates.

## 📖 Documentation

- [DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md) - Complete distributed training guide
- [COST_ANALYSIS.md](COST_ANALYSIS.md) - Training time and cost analysis
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheatsheet

## 🎓 Educational Purpose

This project demonstrates:
- End-to-end ML pipeline design
- Data engineering best practices
- Distributed training implementation
- Music generation with Transformers
- Production-ready code structure

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Music**: miditok, pretty_midi
- **Distributed**: PyTorch DDP, DeepSpeed (optional)
- **Infrastructure**: SLURM support for HPC clusters

## 📝 License

MIT License - feel free to use for educational purposes.

## 🙏 Acknowledgments

- Chopin MIDI files from various public sources
- miditok library for MIDI tokenization
- Hugging Face Transformers library

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating ML engineering skills including data pipeline design, distributed training, and generative AI.
