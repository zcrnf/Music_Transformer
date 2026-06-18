# 3_Transformer.py — Unified Training Script (Single GPU + Distributed)
import os, json, random, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
from miditok import REMI, TokenizerConfig
from torch.optim import AdamW

# ─── settings ─────────────────────────────────────────────────────────────────
SEQ_LEN = 1536
BATCH   = 8
EPOCHS  = 50
LR      = 1e-5
META_JL = Path("metadata_midi.jsonl")

# ─── setup distributed environment ────────────────────────────────────────────
def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# ─── load tokenizer config safely ─────────────────────────────────────────────
with open("tokenizer_config.json") as f:
    cfg_dict = json.load(f)

# ─── build the same config used in 2_Tokenization.py ───────────────
tok_cfg = TokenizerConfig(
    pitch_range      = (21, 108),
    beat_res         = {(0, 4): 4, (4, 8): 8},  # all ints
    use_programs     = True,
    use_tempos       = True,
    num_tempos       = 32,
    num_velocities   = 127,
    tempo_range      = (30, 240),
    additional_tokens= {"Pad": True, "BOS": True, "EOS": True},
)

# ─── make the tokenizer ─────────────────────────────────────────────────
tokenizer = REMI(tok_cfg)
PAD_ID = tokenizer.token_ids_of_type("PAD")[0]
BOS_ID = tokenizer.token_ids_of_type("BOS")[0]
EOS_ID = tokenizer.token_ids_of_type("EOS")[0]

# ─── dataset ──────────────────────────────────────────────────────────────────
class MIDIDataset(Dataset):
    def __init__(self, meta_file: Path, seq_len: int):
        self.seq_len = seq_len
        self.entries = [json.loads(line) for line in meta_file.open()]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        tok_file = Path(self.entries[idx]["tokens"])
        ids = json.load(tok_file.open())["ids"]
        if isinstance(ids[0], list):  # flatten multitrack segments
            ids = [tok for seg in ids for tok in seg]

        toks = torch.tensor(ids, dtype=torch.long)
        need = self.seq_len + 1
        toks = toks[:need] if toks.numel() >= need else \
               torch.cat([toks, torch.full((need - toks.numel(),), PAD_ID)])

        x, y = toks[:-1], toks[1:].clone()
        y[x == PAD_ID] = -100  # ✭ ignore PAD tokens in the loss
        return x, y

# ─── main training function ───────────────────────────────────────────────────
def train(rank=0, world_size=1, distributed=False):
    """Main training function supporting both single GPU and distributed"""
    
    # Setup
    if distributed:
        setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    
    # ─── model ────────────────────────────────────────────────────────────────
    config = GPT2Config(
        vocab_size   = tokenizer.vocab_size,
        n_positions  = SEQ_LEN,
        n_embd       = 1024,
        n_layer      = 18,
        n_head       = 8,
        resid_pdrop  = 0.1,
        attn_pdrop   = 0.1,
        embd_pdrop   = 0.1,
        pad_token_id = PAD_ID,
        bos_token_id = BOS_ID,
        eos_token_id = EOS_ID
    )
    model = GPT2LMHeadModel(config).to(device)
    
    # Wrap with DDP if distributed
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # ─── dataset & dataloader ─────────────────────────────────────────────────
    dataset = MIDIDataset(META_JL, SEQ_LEN)
    
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=BATCH,
            shuffle=True,
            num_workers=2,
            persistent_workers=False
        )
    
    # ─── optimizer & scheduler ────────────────────────────────────────────────
    optim = AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    warmup_steps = int(0.05 * total_steps)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    
    # ─── training loop ────────────────────────────────────────────────────────
    if rank == 0:
        print(f"🚀 Training started...")
        if distributed:
            print(f"   Using {world_size} GPUs")
            print(f"   Effective batch size: {BATCH * world_size}")
        else:
            print(f"   Using single GPU/CPU")
    
    for ep in range(1, EPOCHS + 1):
        model.train()
        if distributed:
            sampler.set_epoch(ep)
        
        total_loss = 0.0
        if rank == 0:
            pbar = tqdm(loader, desc=f"epoch {ep}/{EPOCHS}")
        else:
            pbar = loader
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        # Gather loss from all processes if distributed
        if distributed:
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        if rank == 0:
            print(f"✅ Epoch {ep}: mean loss = {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs (only rank 0)
        if rank == 0 and ep % 10 == 0:
            checkpoint = {
                'epoch': ep,
                'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f"checkpoint_epoch{ep}.pt")
            print(f"💾 Checkpoint saved: epoch {ep}")
    
    # ─── save final model ─────────────────────────────────────────────────────
    if rank == 0:
        model_to_save = model.module if distributed else model
        model_to_save.save_pretrained("saved_model")
        print("🗄️ Model saved to ./saved_model")
    
    if distributed:
        cleanup_distributed()

# ─── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Music Transformer')
    parser.add_argument('--distributed', action='store_true', 
                        help='Enable distributed training (multi-GPU)')
    parser.add_argument('--world-size', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    args = parser.parse_args()
    
    if args.distributed:
        world_size = args.world_size or torch.cuda.device_count()
        
        if world_size < 2:
            print("⚠️ Only 1 GPU available, running single GPU training...")
            train(rank=0, world_size=1, distributed=False)
        else:
            print(f"🎯 Found {world_size} GPUs, starting distributed training...")
            mp.spawn(
                train,
                args=(world_size, True),
                nprocs=world_size,
                join=True
            )
    else:
        # Single GPU/CPU training
        print("Running in single GPU mode (use --distributed for multi-GPU)")
        train(rank=0, world_size=1, distributed=False)
