#!/bin/bash
# launch_distributed.sh - Easy launcher for distributed training

# ============================================================================
# METHOD 1: PyTorch DDP (Manual)
# ============================================================================
# Single node, multiple GPUs (e.g., 4 GPUs on one machine)
launch_ddp_single_node() {
    echo "🚀 Launching DDP training on single node..."
    python scripts/3_Transformer_DDP.py
}

# Multi-node training (e.g., 2 nodes, 4 GPUs each = 8 GPUs total)
launch_ddp_multi_node() {
    NUM_NODES=2
    NUM_GPUS_PER_NODE=4
    NODE_RANK=$1  # 0 for master, 1 for worker
    MASTER_ADDR="192.168.1.100"  # IP of master node
    
    echo "🚀 Launching DDP training on node ${NODE_RANK}..."
    
    torchrun \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=29500 \
        scripts/3_Transformer_DDP.py
}

# ============================================================================
# METHOD 2: Accelerate (Recommended - Easiest)
# ============================================================================
launch_accelerate() {
    echo "🚀 Launching Accelerate training..."
    
    # First time setup (run once):
    # accelerate config
    
    # Run training:
    accelerate launch scripts/3_Transformer_Accelerate.py
}

# Launch on multiple GPUs with Accelerate
launch_accelerate_multi_gpu() {
    echo "🚀 Launching Accelerate training on multiple GPUs..."
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --mixed_precision=fp16 \
        scripts/3_Transformer_Accelerate.py
}

# Multi-node with Accelerate
launch_accelerate_multi_node() {
    NODE_RANK=$1
    NUM_MACHINES=2
    MAIN_PROCESS_IP="192.168.1.100"
    
    accelerate launch \
        --multi_gpu \
        --num_machines=$NUM_MACHINES \
        --num_processes=8 \
        --machine_rank=$NODE_RANK \
        --main_process_ip=$MAIN_PROCESS_IP \
        --main_process_port=29500 \
        scripts/3_Transformer_Accelerate.py
}

# ============================================================================
# METHOD 3: DeepSpeed (For Large Models)
# ============================================================================
launch_deepspeed() {
    echo "🚀 Launching DeepSpeed training..."
    
    deepspeed --num_gpus=4 \
        scripts/3_Transformer_Accelerate.py \
        --deepspeed deepspeed_config.json
}

# ============================================================================
# METHOD 4: SLURM (For HPC Clusters)
# ============================================================================
launch_slurm() {
    echo "🚀 Submitting SLURM job..."
    sbatch slurm_job.sh
}

# ============================================================================
# Default: Run single node multi-GPU with Accelerate
# ============================================================================
main() {
    case "${1:-accelerate}" in
        ddp)
            launch_ddp_single_node
            ;;
        ddp-multi)
            launch_ddp_multi_node ${2:-0}
            ;;
        accelerate)
            launch_accelerate_multi_gpu
            ;;
        accelerate-multi)
            launch_accelerate_multi_node ${2:-0}
            ;;
        deepspeed)
            launch_deepspeed
            ;;
        slurm)
            launch_slurm
            ;;
        *)
            echo "Usage: $0 {ddp|ddp-multi|accelerate|accelerate-multi|deepspeed|slurm} [node_rank]"
            exit 1
            ;;
    esac
}

main "$@"
