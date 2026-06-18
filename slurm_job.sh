#!/bin/bash
#SBATCH --job-name=music_transformer
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --cpus-per-task=8            # CPU cores per task
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --time=24:00:00              # Max runtime (24 hours)
#SBATCH --mem=64G                    # Memory per node
#SBATCH --partition=gpu              # Queue/partition name

# Load modules (adjust for your HPC system)
module purge
module load cuda/11.8
module load python/3.10
module load nccl/2.15

# Activate virtual environment
source ~/venv/bin/activate

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_PER_NODE))

# Get master node hostname
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR

echo "========================================"
echo "SLURM Job Configuration"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $WORLD_SIZE"
echo "Master node: $MASTER_ADDR"
echo "========================================"

# Navigate to project directory
cd /home/zhengmy/ECS111_FP || exit

# Create logs directory
mkdir -p logs

# ============================================================================
# OPTION 1: PyTorch DDP with torchrun
# ============================================================================
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/3_Transformer_DDP.py

# ============================================================================
# OPTION 2: Accelerate (Easier)
# ============================================================================
# srun accelerate launch \
#     --multi_gpu \
#     --num_machines=$SLURM_NNODES \
#     --num_processes=$WORLD_SIZE \
#     --machine_rank=$SLURM_NODEID \
#     --main_process_ip=$MASTER_ADDR \
#     --main_process_port=$MASTER_PORT \
#     scripts/3_Transformer_Accelerate.py

# ============================================================================
# OPTION 3: DeepSpeed (For very large models)
# ============================================================================
# srun deepspeed \
#     --num_nodes=$SLURM_NNODES \
#     --num_gpus=$SLURM_GPUS_PER_NODE \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     scripts/3_Transformer_Accelerate.py \
#     --deepspeed deepspeed_config.json

echo "Training completed!"
