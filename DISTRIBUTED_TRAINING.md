# Distributed Training Guide

This guide explains how to scale your Music Transformer training across multiple GPUs and nodes.

## 📊 Overview

Your training has 3 options, ranked by ease of use:

| Method | Difficulty | Best For | Speed |
|--------|-----------|----------|-------|
| **Accelerate** | ⭐ Easiest | Quick setup, any scale | Fast |
| **PyTorch DDP** | ⭐⭐ Moderate | Full control, custom logic | Fast |
| **DeepSpeed** | ⭐⭐⭐ Advanced | Very large models (>10B params) | Fastest |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Option 1: Accelerate (Recommended)
pip install accelerate

# Option 2: All options
pip install accelerate deepspeed
```

### 2. Run Distributed Training

```bash
# Single machine, multiple GPUs (easiest)
chmod +x launch_distributed.sh
./launch_distributed.sh accelerate

# Or manually:
accelerate launch --multi_gpu --num_processes=4 scripts/3_Transformer_Accelerate.py
```

---

## 📚 Detailed Setup

### Option 1: Accelerate (Recommended) ⭐

**Why:** Minimal code changes, handles all distributed logic automatically.

#### Setup (One-time):
```bash
accelerate config
```

Answer the prompts:
- Multi-GPU training: Yes
- How many machines: 1 (for single node)
- Number of processes: 4 (number of GPUs)
- Mixed precision: fp16
- DeepSpeed: No (unless you need it)

#### Run Training:
```bash
# Single node, auto-detect GPUs
accelerate launch scripts/3_Transformer_Accelerate.py

# Specify number of GPUs
accelerate launch --multi_gpu --num_processes=4 scripts/3_Transformer_Accelerate.py

# Multi-node (run on each machine)
# Machine 0 (master):
accelerate launch --multi_gpu --num_machines=2 --machine_rank=0 \
    --main_process_ip=192.168.1.100 --main_process_port=29500 \
    --num_processes=8 scripts/3_Transformer_Accelerate.py

# Machine 1 (worker):
accelerate launch --multi_gpu --num_machines=2 --machine_rank=1 \
    --main_process_ip=192.168.1.100 --main_process_port=29500 \
    --num_processes=8 scripts/3_Transformer_Accelerate.py
```

---

### Option 2: PyTorch DDP ⭐⭐

**Why:** More control, no extra dependencies beyond PyTorch.

#### Single Node (Multiple GPUs):
```bash
# Auto-detects all GPUs
python scripts/3_Transformer_DDP.py

# Using torchrun (more flexible)
torchrun --nproc_per_node=4 scripts/3_Transformer_DDP.py
```

#### Multi-Node:
```bash
# Node 0 (master) - IP: 192.168.1.100
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 \
    scripts/3_Transformer_DDP.py

# Node 1 (worker) - IP: 192.168.1.101
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=29500 \
    scripts/3_Transformer_DDP.py
```

---

### Option 3: DeepSpeed ⭐⭐⭐

**Why:** Memory-efficient training for very large models (ZeRO optimization).

#### Setup:
```bash
pip install deepspeed
```

#### Run:
```bash
# Single node
deepspeed --num_gpus=4 scripts/3_Transformer_Accelerate.py \
    --deepspeed deepspeed_config.json

# Multi-node (create hostfile first)
echo "node1 slots=4" > hostfile
echo "node2 slots=4" >> hostfile

deepspeed --hostfile=hostfile scripts/3_Transformer_Accelerate.py \
    --deepspeed deepspeed_config.json
```

**DeepSpeed Features:**
- `stage: 1` - Optimizer state partitioning (1.5x memory savings)
- `stage: 2` - Optimizer + gradients partitioning (3x memory savings)
- `stage: 3` - Optimizer + gradients + parameters (10x+ memory savings)

Edit `deepspeed_config.json` to change ZeRO stage.

---

## 🖥️ HPC/SLURM Clusters

If you're using an HPC cluster with SLURM:

```bash
# Submit job
sbatch slurm_job.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/train_<jobid>.out
```

Edit `slurm_job.sh` to adjust:
- `--nodes`: Number of nodes
- `--gres=gpu:X`: GPUs per node
- `--time`: Max runtime
- Module names (depends on your cluster)

---

## 📈 Performance Tips

### Batch Size
- Start with `BATCH=8` per GPU
- Effective batch size = `BATCH × num_GPUs × gradient_accumulation_steps`
- Larger batch sizes usually train faster but may reduce model quality

### Gradient Accumulation
Simulate larger batches without more memory:

```python
# In Accelerate script
accelerator = Accelerator(gradient_accumulation_steps=4)
```

Now effective batch = 8 × 4 GPUs × 4 = 128

### Mixed Precision (FP16/BF16)
Speeds up training by 2-3x:

```python
accelerator = Accelerator(mixed_precision='fp16')  # or 'bf16'
```

### Profiling
Find bottlenecks:

```bash
# PyTorch Profiler
python -m torch.distributed.run --nproc_per_node=4 \
    scripts/3_Transformer_DDP.py --profile
```

---

## 🔧 Troubleshooting

### Error: "NCCL timeout"
**Solution:** Increase timeout
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

### Error: "Connection refused"
**Solution:** Check firewall, ensure master node is reachable
```bash
# On worker nodes
ping 192.168.1.100  # master IP
telnet 192.168.1.100 29500  # check port
```

### Error: "CUDA out of memory"
**Solutions:**
1. Reduce batch size: `BATCH = 4`
2. Enable gradient accumulation
3. Use DeepSpeed ZeRO stage 2 or 3
4. Enable gradient checkpointing (see below)

```python
# Gradient checkpointing (saves memory)
model.gradient_checkpointing_enable()
```

### Training is slow
**Check:**
1. GPU utilization: `nvidia-smi dmon`
2. Data loading: increase `num_workers`
3. Network bandwidth (multi-node): `iperf3`

---

## 📊 Expected Speedups

| Setup | GPUs | Time per Epoch | Speedup |
|-------|------|----------------|---------|
| Single GPU | 1 | ~60 min | 1x |
| Single Node | 4 | ~17 min | 3.5x |
| Multi-Node | 8 | ~9 min | 6.7x |
| Multi-Node | 16 | ~5 min | 12x |

*Note: Actual speedups depend on model size, batch size, and network bandwidth*

---

## 📝 Resume Training

Add to your script for checkpointing:

```python
# Save checkpoint
if rank == 0 and epoch % 10 == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, f'checkpoint_epoch{epoch}.pt')

# Resume from checkpoint
if os.path.exists('checkpoint_latest.pt'):
    checkpoint = torch.load('checkpoint_latest.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

---

## 🎯 Which Method Should I Use?

| Scenario | Recommended Method |
|----------|-------------------|
| Just getting started | **Accelerate** |
| 1-4 GPUs on one machine | **Accelerate** or **PyTorch DDP** |
| Multiple machines | **Accelerate** (easiest) or **PyTorch DDP** |
| Model >10B parameters | **DeepSpeed ZeRO-3** |
| Need fine control | **PyTorch DDP** |
| Using HPC cluster | **SLURM + Accelerate** |

---

## 📚 Additional Resources

- [Accelerate Docs](https://huggingface.co/docs/accelerate)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DeepSpeed Docs](https://www.deepspeed.ai/getting-started/)
- [NCCL Performance Tuning](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
