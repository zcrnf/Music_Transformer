#!/usr/bin/env python3
"""
Memory calculator for MIDI transformer training
"""

def calculate_memory(seq_len, vocab_size, d_model, n_layers, n_heads, batch_size, Q=1):
    """
    Calculate approximate GPU memory for transformer training.
    
    For MIDI (single-stream):
        Q = 1 (single token stream, not multi-codebook)
    For Audio (multi-codebook):
        Q = 8 (8 codebook streams)
    """
    
    # Model parameters
    # Embeddings: vocab_size * d_embed (assuming d_embed = d_model for single-stream)
    embedding_params = vocab_size * d_model * Q
    
    # Transformer blocks: roughly 12 * d_model^2 per layer
    # (QKV projections, FFN, layer norms)
    transformer_params = n_layers * 12 * (d_model ** 2)
    
    # Output heads: d_model * vocab_size per codebook
    head_params = d_model * vocab_size * Q
    
    total_params = embedding_params + transformer_params + head_params
    
    # Memory breakdown (in GB)
    # 1. Model weights (FP32)
    model_memory = (total_params * 4) / (1024**3)
    
    # 2. Optimizer states (AdamW = 2x params for momentum + variance)
    optimizer_memory = (total_params * 8) / (1024**3)
    
    # 3. Gradients
    gradient_memory = (total_params * 4) / (1024**3)
    
    # 4. Activations (largest component for long sequences)
    # Rough estimate: batch_size * seq_len * n_layers * d_model * 4 (attention maps)
    activation_memory = (batch_size * seq_len * n_layers * d_model * 16) / (1024**3)
    
    # 5. Mixed precision (AMP) - saves ~40% on activations/weights
    amp_savings = (model_memory + activation_memory) * 0.4
    
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory - amp_savings
    
    # Add 20% overhead for PyTorch/CUDA
    total_memory *= 1.2
    
    return {
        'params_M': total_params / 1e6,
        'model_GB': model_memory,
        'optimizer_GB': optimizer_memory,
        'gradients_GB': gradient_memory,
        'activations_GB': activation_memory,
        'amp_savings_GB': amp_savings,
        'total_GB': total_memory
    }

# Your current audio architecture (from 4_Transformer.py)
print("="*70)
print("CURRENT AUDIO ARCHITECTURE (Multi-codebook)")
print("="*70)
audio_config = {
    'seq_len': 600,      # frames (4 seconds at 150 fps)
    'vocab_size': 1026,  # Encodec codebook size
    'd_model': 512,
    'n_layers': 8,
    'n_heads': 8,
    'batch_size': 4,
    'Q': 8               # 8 codebooks
}

result = calculate_memory(**audio_config)
print(f"Parameters: {result['params_M']:.1f}M")
print(f"Model:      {result['model_GB']:.2f} GB")
print(f"Optimizer:  {result['optimizer_GB']:.2f} GB")
print(f"Gradients:  {result['gradients_GB']:.2f} GB")
print(f"Activations:{result['activations_GB']:.2f} GB")
print(f"AMP savings:{result['amp_savings_GB']:.2f} GB")
print(f"TOTAL:      {result['total_GB']:.2f} GB\n")

# MIDI architectures to test
print("="*70)
print("MIDI SINGLE-STREAM ARCHITECTURES (Q=1)")
print("="*70)

configs = [
    # Conservative (fits 40GB easily)
    {'name': '1K context, batch=8', 'seq_len': 1024, 'vocab_size': 500, 'd_model': 512, 'n_layers': 8, 'n_heads': 8, 'batch_size': 8, 'Q': 1},
    {'name': '2K context, batch=4', 'seq_len': 2048, 'vocab_size': 500, 'd_model': 512, 'n_layers': 8, 'n_heads': 8, 'batch_size': 4, 'Q': 1},
    
    # Medium (good for A100 40GB)
    {'name': '4K context, batch=4', 'seq_len': 4096, 'vocab_size': 500, 'd_model': 512, 'n_layers': 8, 'n_heads': 8, 'batch_size': 4, 'Q': 1},
    {'name': '4K context, batch=2', 'seq_len': 4096, 'vocab_size': 500, 'd_model': 512, 'n_layers': 8, 'n_heads': 8, 'batch_size': 2, 'Q': 1},
    
    # Large (for A100 80GB)
    {'name': '8K context, batch=2', 'seq_len': 8192, 'vocab_size': 500, 'd_model': 512, 'n_layers': 8, 'n_heads': 8, 'batch_size': 2, 'Q': 1},
    {'name': '8K context, batch=4', 'seq_len': 8192, 'vocab_size': 500, 'd_model': 512, 'n_layers': 8, 'n_heads': 8, 'batch_size': 4, 'Q': 1},
    
    # Deeper model (better quality)
    {'name': '4K, 12 layers, batch=2', 'seq_len': 4096, 'vocab_size': 500, 'd_model': 512, 'n_layers': 12, 'n_heads': 8, 'batch_size': 2, 'Q': 1},
    {'name': '4K, d=768, batch=2', 'seq_len': 4096, 'vocab_size': 500, 'd_model': 768, 'n_layers': 8, 'n_heads': 12, 'batch_size': 2, 'Q': 1},
]

print(f"{'Config':<30} {'Params':<12} {'Memory (GB)':<12} {'Fits 40GB?':<12} {'Fits 80GB?'}")
print("-"*70)

for cfg in configs:
    name = cfg.pop('name')
    result = calculate_memory(**cfg)
    fits_40 = "✅ YES" if result['total_GB'] < 38 else "❌ NO"
    fits_80 = "✅ YES" if result['total_GB'] < 75 else "❌ NO"
    print(f"{name:<30} {result['params_M']:>6.1f}M     {result['total_GB']:>6.2f} GB    {fits_40:<12} {fits_80}")

print("="*70)
print("\n💡 RECOMMENDATIONS:")
print("   A100 40GB: Use seq_len=4096, batch_size=2-4 (with gradient accumulation)")
print("   A100 80GB: Use seq_len=8192, batch_size=2-4")
print("\n   With gradient accumulation x4, effective batch = batch_size * 4")
print("   E.g., batch=2, accum=4 → effective batch=8")
print("\n   Note: Actual memory may vary ±20% depending on implementation details")
