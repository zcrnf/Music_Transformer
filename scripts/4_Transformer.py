# 4_Transformer_train_or_resume.py — Train *or* Resume in one script
# GPT2-style model on Encodec Q-codebook tokens with separate heads
# Features: weight tying, PAD masking, random crops, AMP, grad clip,
# cosine LR warmup, accumulation, TF32, torch.compile, prefix-strip
# when resuming, and clean unwrap when saving.

import json
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Config, GPT2Model, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# ─────────── Repro & Speed ───────────

def seed_all(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_all()

# Prefer TF32 for speed (safe for training quality)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# Special token IDs from tokenizer (miditok REMI)
PAD_ID     = 0   # PAD_None
BOS_ID     = 1   # BOS_None
EOS_ID     = 2   # EOS_None
VOCAB_SIZE = 419  # Full MIDI vocab size (0-418)
IGNORE_INDEX = -100
COMPOSER_SCALE = 0.25  # Scale factor for composer conditioning (gentle bias)

# ─────────── Helpers ───────────

def infer_seq_len(encoded_dir: str) -> int:
    """Return sample length from any token file under encoded_dir."""
    encoded_path = Path(encoded_dir)
    token_files  = list(encoded_path.rglob("*.th"))
    if not token_files:
        raise FileNotFoundError(f"No .th token files found in {encoded_dir}")
    tokens = torch.load(token_files[0], map_location="cpu")  # [T] single-stream
    if tokens.ndim != 1:
        raise ValueError(f"Expected token shape [T], got {tokens.shape}")
    T = tokens.shape[0]
    print(f"🔍 Found sample length T={T} in {token_files[0].name}")
    return T

class MIDITokenDataset(Dataset):
    def __init__(self, meta_path: Path, seq_len: int, cfg_dropout: float = 0.6):
        with open(meta_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f]
        self.seq_len = seq_len
        self.cfg_dropout = cfg_dropout  # Probability of unconditional training
        self.num_composers = 238  # 237 frequent composers + 1 OTHER (0 reserved for unconditional)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        tpath = Path(row["token_path"])  # path to .th
        tokens = torch.load(tpath, map_location="cpu").long()  # [T] single-stream

        # Safety: handle empty or invalid tokens
        if tokens.numel() == 0 or tokens.ndim != 1:
            # Fallback: create minimal valid sequence
            tokens = torch.tensor([BOS_ID, PAD_ID], dtype=torch.long)

        # random crop BEFORE BOS/PAD
        T = tokens.shape[0]
        crop = self.seq_len
        if T > crop:
            start = torch.randint(0, T - crop + 1, (1,)).item()
            tokens = tokens[start:start+crop]

        # tiny denoising: drop 1% tokens to PAD
        if tokens.numel() > 0:
            mask = torch.rand(tokens.size(0)) < 0.01
            tokens[mask] = PAD_ID

        # prepend BOS and pad/trim to (seq_len+1)
        bos = torch.tensor([BOS_ID], dtype=torch.long)
        tokens = torch.cat([bos, tokens], 0)  # [L+1]

        need = self.seq_len + 1
        if tokens.shape[0] < need:
            pad = torch.full((need - tokens.shape[0],), PAD_ID, dtype=torch.long)
            tokens = torch.cat([tokens, pad], 0)
        else:
            tokens = tokens[:need]

        input_ids = tokens[:-1]
        labels    = tokens[1:].clone()
        labels[labels == PAD_ID] = IGNORE_INDEX
        
        # Get composer_id from metadata (clamp to valid range)
        composer_id = row.get('composer_id', 1)
        composer_id = max(0, min(composer_id, self.num_composers if hasattr(self, 'num_composers') else 2567))
        
        # Classifier-free guidance: 10% chance to drop conditioning (use 0 = unconditional)
        if random.random() < self.cfg_dropout:
            composer_id = 0
        
        return {
            "input_ids": input_ids, 
            "labels": labels,
            "composer_id": torch.tensor(composer_id, dtype=torch.long)
        }

class MIDITransformer(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, n_layer, n_head, num_composers=2567):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_composers = num_composers

        # Single embedding layer for MIDI tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        
        # Composer conditioning embedding (index 0 reserved for unconditional/null)
        # No padding_idx here since 0 is a valid conditioning value
        self.composer_embedding = nn.Embedding(num_composers + 1, d_model)
        
        self.dropout = nn.Dropout(0.1)

        cfg = GPT2Config(
            vocab_size=1,
            n_positions=seq_len,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.transformer = GPT2Model(cfg)
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()

        # Single output head with weight tying
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Scale factor for logits (important for weight tying)
        self.logit_scale = d_model ** -0.5

    def _compute_attn_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != PAD_ID).long()

    def forward(self, input_ids: torch.Tensor, composer_id: torch.Tensor = None, labels: torch.Tensor = None):
        B, T = input_ids.shape

        # Default to unconditional if not provided
        if composer_id is None:
            composer_id = torch.zeros(B, dtype=torch.long, device=input_ids.device)

        # Token embeddings + composer conditioning (scaled to act as gentle bias)
        x = self.token_embedding(input_ids)  # [B, T, d_model]
        comp_emb = self.composer_embedding(composer_id).unsqueeze(1)  # [B, 1, d_model]
        x = x + COMPOSER_SCALE * comp_emb  # Broadcast scaled composer embedding across time
        x = self.dropout(x)

        hidden = self.transformer(
            inputs_embeds=x,
            attention_mask=self._compute_attn_mask(input_ids)
        ).last_hidden_state

        logits = self.lm_head(hidden) * self.logit_scale  # Scale logits for stability

        if labels is None:
            return logits

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=0.1,
        )
        return loss

# --- prefix helpers for resuming compiled/DDP checkpoints ---

def strip_prefixes(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."): k = k[10:]
        if k.startswith("module."):   k = k[7:]
        out[k] = v
    return out

def unwrap_for_saving(model: nn.Module) -> nn.Module:
    if hasattr(model, "_orig_mod"):  # torch.compile wrapper
        return model._orig_mod
    if hasattr(model, "module"):     # DDP
        return model.module
    return model

# ─────────── Build optimizer/scheduler/dataloader ───────────

def make_loader(meta_path: Path, seq_len: int, batch_size: int, device: torch.device, cfg_dropout: float = 0.6):
    ds = MIDITokenDataset(meta_path, seq_len, cfg_dropout=cfg_dropout)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    return ds, dl

# ─────────── Train Loop ───────────

def train_loop(model, dl, opt, scheduler, scaler, device, accum_steps, max_norm, start_ep, end_ep, out_dir, meta, log_file):
    print("🚀  Starting training …", flush=True)
    for ep in range(start_ep + 1, end_ep + 1):
        model.train()
        running_loss, step = 0.0, 0
        opt.zero_grad(set_to_none=True)
        
        errors = 0
        max_errors = 10

        pbar = tqdm(dl, desc=f"Epoch {ep}/{end_ep}", ncols=100, file=sys.stdout)
        for batch in pbar:
            try:
                inp  = batch["input_ids"].to(device)
                labs = batch["labels"].to(device)
                comp_id = batch["composer_id"].to(device)
                
                if torch.isnan(inp).any() or torch.isinf(inp).any():
                    errors += 1
                    if errors <= max_errors:
                        continue
                    else:
                        raise RuntimeError("Too many NaN/Inf batches, stopping training")

                with autocast(enabled=(device.type == "cuda")):
                    loss = model(inp, composer_id=comp_id, labels=labs) / accum_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    errors += 1
                    tqdm.write(f"⚠️  Warning: NaN/Inf loss detected, skipping batch {step}")
                    if errors > max_errors:
                        raise RuntimeError("Too many NaN losses, stopping training")
                    continue

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()

                running_loss += loss.item() * accum_steps
                step += 1
                
                # Update tqdm with real-time loss
                if step > 0:
                    current_avg = running_loss / step
                    pbar.set_postfix({'loss': f'{current_avg:.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    tqdm.write(f"❌ GPU OOM at step {step}, clearing cache and continuing...")
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    errors += 1
                    if errors > max_errors:
                        raise
                    continue
                else:
                    raise

        if step == 0:
            msg = f"❌ Epoch {ep}: No valid batches processed!"
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()
            continue

        avg = running_loss / step
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to file ONCE per epoch
        log_msg = f"Epoch {ep}/{end_ep} | Avg Loss: {avg:.4f} | LR: {current_lr:.2e} | Errors: {errors}\n"
        log_file.write(log_msg)
        log_file.flush()

        # Save checkpoint every epoch
        to_save = unwrap_for_saving(model)
        ckpt = {
            **meta,
            "epoch": ep,
            "model_state_dict": to_save.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg,
        }
        path = out_dir / f"music_transformer_ep{ep}.pt"
        torch.save(ckpt, path)
        save_msg = f"📦  Saved checkpoint to {path.resolve()}"
        print(save_msg, flush=True)
        log_file.write(save_msg + "\n")
        log_file.flush()
        
        # Copy to Windows WSL path
        try:
            wsl_path = Path("/mnt/c/Users") / "zhengmy" / "Transformer_ECS111"
            wsl_path.mkdir(parents=True, exist_ok=True)
            wsl_dest = wsl_path / f"music_transformer_ep{ep}.pt"
            shutil.copy2(path, wsl_dest)
            copy_msg = f"📋  Copied checkpoint to {wsl_dest}"
            print(copy_msg, flush=True)
            log_file.write(copy_msg + "\n")
            log_file.flush()
        except Exception as e:
            err_msg = f"⚠️  Failed to copy to Windows path: {e}"
            print(err_msg, flush=True)
            log_file.write(err_msg + "\n")
            log_file.flush()

    final_msg = "🎉  Training complete."
    print(final_msg, flush=True)
    log_file.write(final_msg + "\n")
    log_file.flush()

# ─────────── Main ───────────
if __name__ == "__main__":
    OUTPUT_DIR  = Path("model_results_midis"); OUTPUT_DIR.mkdir(exist_ok=True)
    META_PATH   = Path("metadata_clean_midis.jsonl")
    TOKEN_ROOT  = Path("encoded_tokens/midis")
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base hyperparams
    default_seq = 8192  # default context window

    mode = "resume"  # Auto-start new training
    
    if mode == "new":
        # NEW training
        SEQ_LEN   = 8192
        D_MODEL   = 512
        N_LAYER   = 8
        N_HEAD    = 8
        BATCH_SIZE= 2
        ACCUM_STEPS = 8
        MAX_NORM  = 1.0
        WARMUP_FRAC = 0.05
        WEIGHT_DECAY = 0.01
        EPOCHS    = 500
        LR        = 5e-4

        # build model
        NUM_COMPOSERS = 238  # 237 frequent composers + 1 OTHER (ID 0 reserved for unconditional)
        model = MIDITransformer(VOCAB_SIZE, D_MODEL, SEQ_LEN, N_LAYER, N_HEAD, num_composers=NUM_COMPOSERS).to(DEVICE)
        try:
            model = torch.compile(model)
        except Exception:
            pass

        # data/opt/sched
        ds, dl = make_loader(META_PATH, SEQ_LEN, BATCH_SIZE, DEVICE, cfg_dropout=0.6)
        opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), eps=1e-8)
        total_steps = max(1, (len(dl) * EPOCHS) // ACCUM_STEPS)
        warmup = max(1, int(WARMUP_FRAC * total_steps))
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)
        scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

        meta = {
            'd_model': D_MODEL, 'seq_len': SEQ_LEN,
            'n_layer': N_LAYER, 'n_head': N_HEAD, 'vocab_size': VOCAB_SIZE,
            'num_composers': NUM_COMPOSERS
        }
        
        log_path = Path("training.log")
        with open(log_path, "w") as log_file:
            train_loop(model, dl, opt, scheduler, scaler, DEVICE, ACCUM_STEPS, MAX_NORM, 0, EPOCHS, OUTPUT_DIR, meta, log_file)

    elif mode == "resume":
        # RESUME training
        start_ep = int(input("Resume from epoch (e.g. 100): "))
        target_ep = int(input("Target epoch to reach (e.g. 600): "))
        LR        = float(input("Learning rate for resume (e.g. 5e-4): ") or 5e-4)
        BATCH_SIZE= 4
        ACCUM_STEPS = 4
        MAX_NORM  = 1.0
        WARMUP_FRAC = 0.05
        WEIGHT_DECAY = 0.01

        ckpt_path = OUTPUT_DIR / f"music_transformer_ep{start_ep}.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        D_MODEL = ckpt['d_model']
        SEQ_LEN = ckpt['seq_len']
        N_LAYER = ckpt['n_layer']
        N_HEAD  = ckpt['n_head']
        NUM_COMPOSERS = ckpt.get('num_composers', 2567)

        model = MIDITransformer(VOCAB_SIZE, D_MODEL, SEQ_LEN, N_LAYER, N_HEAD, num_composers=NUM_COMPOSERS).to(DEVICE)
        model.load_state_dict(strip_prefixes(ckpt['model_state_dict']), strict=True)
        try:
            model = torch.compile(model)
        except Exception:
            pass

        ds, dl = make_loader(META_PATH, SEQ_LEN, BATCH_SIZE, DEVICE)
        opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), eps=1e-8)
        try:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠️ Could not load optimizer state: {e}. Using fresh optimizer.")

        steps_left = max(1, (len(dl) * (target_ep - start_ep)) // ACCUM_STEPS)
        warmup = max(1, int(WARMUP_FRAC * steps_left))
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps_left)
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                print(f"⚠️ Could not load scheduler state: {e}. Using fresh scheduler.")
        scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

        meta = {
            'd_model': D_MODEL, 'seq_len': SEQ_LEN,
            'n_layer': N_LAYER, 'n_head': N_HEAD, 'vocab_size': VOCAB_SIZE,
            'num_composers': NUM_COMPOSERS
        }
        
        log_path = Path("training.log")
        with open(log_path, "a") as log_file:
            train_loop(model, dl, opt, scheduler, scaler, DEVICE, ACCUM_STEPS, MAX_NORM, start_ep, target_ep, OUTPUT_DIR, meta, log_file)

    else:
        raise SystemExit("Please type 'new' or 'resume'.")
