# 🚀 Distributed Training Quick Reference

## Installation
```bash
pip install accelerate
# Optional: pip install deepspeed
```

## Single Command Quick Start

```bash
# 🌟 EASIEST: Accelerate (auto-detect all GPUs)
accelerate launch scripts/3_Transformer_Accelerate.py

# 🔧 PyTorch DDP (auto-detect all GPUs)
python scripts/3_Transformer_DDP.py

# 🎯 Using launch script
./launch_distributed.sh accelerate
```

## Common Commands

### Accelerate
```bash
# Setup (first time)
accelerate config

# Single node, 4 GPUs
accelerate launch --multi_gpu --num_processes=4 scripts/3_Transformer_Accelerate.py

# Multi-node (on each machine)
accelerate launch --multi_gpu --num_machines=2 --machine_rank=0 \
    --main_process_ip=MASTER_IP --num_processes=8 scripts/3_Transformer_Accelerate.py
```

### PyTorch DDP
```bash
# Single node
torchrun --nproc_per_node=4 scripts/3_Transformer_DDP.py

# Multi-node
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=MASTER_IP --master_port=29500 scripts/3_Transformer_DDP.py
```

### DeepSpeed
```bash
deepspeed --num_gpus=4 scripts/3_Transformer_Accelerate.py --deepspeed deepspeed_config.json
```

### SLURM
```bash
sbatch slurm_job.sh
squeue -u $USER
tail -f logs/train_*.out
```

## Monitoring

```bash
# GPU usage (real-time)
nvidia-smi dmon

# Process monitoring
watch -n 1 nvidia-smi

# Network monitoring (multi-node)
iftop -i eth0
```

## Troubleshooting

```bash
# Out of memory → reduce batch size
BATCH=4 python scripts/3_Transformer_Accelerate.py

# NCCL timeout → increase timeout
export NCCL_TIMEOUT=1800

# Check connectivity (multi-node)
ping MASTER_IP
telnet MASTER_IP 29500
```

## Environment Variables

```bash
# Increase timeouts
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1

# Debug mode
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Optimize NCCL
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # Network interface
```

## File Overview

| File | Purpose |
|------|---------|
| `3_Transformer_DDP.py` | PyTorch DDP implementation |
| `3_Transformer_Accelerate.py` | Accelerate implementation (easiest) |
| `launch_distributed.sh` | Launch script for all methods |
| `deepspeed_config.json` | DeepSpeed configuration |
| `slurm_job.sh` | SLURM job script |
| `accelerate_config.yaml` | Accelerate config template |
| `DISTRIBUTED_TRAINING.md` | Full documentation |

## Resume Checklist

When describing this on your resume:

✅ "Implemented distributed training pipeline scaling to X GPUs"
✅ "Reduced training time from X hours to Y minutes using multi-GPU training"
✅ "Optimized data parallelism with PyTorch DDP/Accelerate"
✅ "Deployed training on HPC cluster using SLURM"
✅ "Utilized mixed-precision training (FP16) for 2x speedup"
✅ "Managed gradient accumulation for effective batch sizes of XYZ"

## Quick Decision Tree

```
Need distributed training?
│
├─ Yes, just starting → Use Accelerate
├─ Yes, need control → Use PyTorch DDP  
├─ Model >10B params → Use DeepSpeed
├─ On HPC cluster → Use SLURM + Accelerate
└─ Single GPU → Use original 3_Transformer.py
```
