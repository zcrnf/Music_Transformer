# 4_Transformer_train_or_resume.py — Train *or* Resume in one script
# GPT2-style model on Encodec Q-codebook tokens with separate heads
# Features: weight tying, PAD masking, random crops, AMP, grad clip,
# cosine LR warmup, accumulation, TF32, torch.compile, prefix-strip
# when resuming, and clean unwrap when saving.

import json
import random
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

# ─────────── Specials ───────────
BOS_ID  = 1024
PAD_ID  = 1025
VOCAB_SIZE = PAD_ID + 1
IGNORE_INDEX = -100

# ─────────── Helpers ───────────

def infer_seq_len(encoded_dir: str) -> tuple[int, int]:
    """Return (T, Q) from any token file under encoded_dir."""
    encoded_path = Path(encoded_dir)
    token_files  = list(encoded_path.rglob("*.th"))
    if not token_files:
        raise FileNotFoundError(f"No .th token files found in {encoded_dir}")
    obj = torch.load(token_files[0], map_location="cpu")
    codes = obj.get("codes", obj) if isinstance(obj, dict) else obj   # [1,Q,T]
    if codes.ndim != 3:
        raise ValueError(f"Expected token shape [1, Q, T], got {codes.shape}")
    _, Q, T = codes.shape
    print(f"🔍 Found Q={Q}, sample length T={T} in {token_files[0].name}")
    return T, Q

class EncodecTokenDataset(Dataset):
    def __init__(self, meta_path: Path, seq_len: int, Q: int):
        with open(meta_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f]
        self.seq_len = seq_len
        self.Q = Q

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        tpath = Path(row["audio"])  # path to .th
        obj = torch.load(tpath, map_location="cpu")
        codes = obj.get("codes", obj) if isinstance(obj, dict) else obj  # [1,Q,T]
        tokens = codes[0].transpose(0, 1).long()                           # [T,Q]

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
        bos = torch.full((1, self.Q), BOS_ID, dtype=torch.long)
        tokens = torch.cat([bos, tokens], 0)  # [L+1,Q]

        need = self.seq_len + 1
        if tokens.shape[0] < need:
            pad = torch.full((need - tokens.shape[0], self.Q), PAD_ID, dtype=torch.long)
            tokens = torch.cat([tokens, pad], 0)
        else:
            tokens = tokens[:need]

        input_ids = tokens[:-1]
        labels    = tokens[1:].clone()
        labels[labels == PAD_ID] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels}

class CodebookMultiHeadTransformer(nn.Module):
    def __init__(self, vocab_size, d_embed, Q, d_model, seq_len, n_layer, n_head):
        super().__init__()
        self.Q = Q
        self.vocab_size = vocab_size
        self.d_embed = d_embed

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_embed, padding_idx=PAD_ID) for _ in range(Q)
        ])
        self.linear_proj = nn.Linear(Q * d_embed, d_model)

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

        self.heads = nn.ModuleList([
            nn.Linear(d_model, d_embed, bias=False) for _ in range(Q)
        ])
        self.dropout = nn.Dropout(0.1)

    def _compute_attn_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids[..., 0] != PAD_ID).long()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        B, T, Q = input_ids.shape
        assert Q == self.Q

        embeds = [self.embeddings[q](input_ids[:, :, q]) for q in range(Q)]
        concat = torch.cat(embeds, dim=-1)
        x = self.linear_proj(concat)
        x = self.dropout(x)

        hidden = self.transformer(
            inputs_embeds=x,
            attention_mask=self._compute_attn_mask(input_ids)
        ).last_hidden_state

        if labels is None:
            logits = []
            for q in range(Q):
                z = self.heads[q](hidden)
                logits_q = torch.matmul(z, self.embeddings[q].weight.T)
                logits.append(logits_q)
            return logits

        total = 0.0
        for q in range(Q):
            z = self.heads[q](hidden)
            logits_q = torch.matmul(z, self.embeddings[q].weight.T)
            loss_q = F.cross_entropy(
                logits_q.reshape(-1, self.vocab_size),
                labels[:, :, q].reshape(-1),
                ignore_index=IGNORE_INDEX,
                label_smoothing=0.1,
            )
            total = total + loss_q
        return total / Q

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

def make_loader(meta_path: Path, seq_len: int, Q: int, batch_size: int, device: torch.device):
    ds = EncodecTokenDataset(meta_path, seq_len, Q)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    return ds, dl

# ─────────── Train Loop ───────────

def train_loop(model, dl, opt, scheduler, scaler, device, accum_steps, max_norm, start_ep, end_ep, out_dir, meta):
    print("🚀  Starting training …")
    for ep in range(start_ep + 1, end_ep + 1):
        model.train()
        running_loss, step = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for batch in tqdm(dl):
            inp  = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)

            with autocast(enabled=(device.type == "cuda")):
                loss = model(inp, labels=labs) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * accum_steps
            step += 1

        avg = running_loss / step
        print(f"✅  Epoch {ep}/{end_ep} — Avg Loss: {avg:.4f}")

        # save cleanly (unwrap compiled/DDP)
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
        print(f"📦  Saved checkpoint to {path.resolve()}")

    print("🎉  Training complete.")

# ─────────── Main ───────────
if __name__ == "__main__":
    folder_name = input("Enter the name of artist: ")
    OUTPUT_DIR  = Path(f"model_results_{folder_name}"); OUTPUT_DIR.mkdir(exist_ok=True)
    META_PATH   = Path(f"metadata_clean_{folder_name}.jsonl")
    TOKEN_ROOT  = Path(f"encoded_tokens/{folder_name}")
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base hyperparams
    TOKENS_PER_SEC = 150
    # infer Q and initial T to propose seq_len for NEW training
    try:
        T_infer, Q_infer = infer_seq_len(str(TOKEN_ROOT))
    except Exception as e:
        print(f"⚠️ Could not infer from tokens yet: {e}")
        T_infer, Q_infer = 600, 8  # sensible defaults

    default_seq = min(T_infer, TOKENS_PER_SEC * 4)  # ≈4s

    mode = input("Type 'new' to start fresh, or 'resume' to continue training: ").strip().lower()

    if mode == "new":
        # NEW training
        SEQ_LEN   = int(input(f"Context length (frames) [default {default_seq}]: ") or default_seq)
        Q         = Q_infer
        D_EMBED   = 64
        D_MODEL   = Q * D_EMBED
        N_LAYER   = 8
        N_HEAD    = 8
        BATCH_SIZE= 5
        ACCUM_STEPS = 4
        MAX_NORM  = 1.0
        WARMUP_FRAC = 0.05
        WEIGHT_DECAY = 0.01
        EPOCHS    = int(input("Total epochs to train (e.g. 500): ") or 500)
        LR        = float(input("Learning rate (e.g. 5e-4): ") or 5e-4)

        # build model
        model = CodebookMultiHeadTransformer(VOCAB_SIZE, D_EMBED, Q, D_MODEL, SEQ_LEN, N_LAYER, N_HEAD).to(DEVICE)
        try:
            model = torch.compile(model)
        except Exception:
            pass

        # data/opt/sched
        ds, dl = make_loader(META_PATH, SEQ_LEN, Q, BATCH_SIZE, DEVICE)
        opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), eps=1e-8)
        total_steps = max(1, (len(dl) * EPOCHS) // ACCUM_STEPS)
        warmup = max(1, int(WARMUP_FRAC * total_steps))
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)
        scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

        meta = {
            'Q': Q, 'd_embed': D_EMBED, 'd_model': D_MODEL, 'seq_len': SEQ_LEN,
            'n_layer': N_LAYER, 'n_head': N_HEAD, 'vocab_size': VOCAB_SIZE
        }
        train_loop(model, dl, opt, scheduler, scaler, DEVICE, ACCUM_STEPS, MAX_NORM, 0, EPOCHS, OUTPUT_DIR, meta)

    elif mode == "resume":
        # RESUME training
        start_ep = int(input("Resume from epoch (e.g. 100): "))
        target_ep = int(input("Target epoch to reach (e.g. 600): "))
        LR        = float(input("Learning rate for resume (e.g. 5e-4): ") or 5e-4)
        BATCH_SIZE= 5
        ACCUM_STEPS = 4
        MAX_NORM  = 1.0
        WARMUP_FRAC = 0.05
        WEIGHT_DECAY = 0.01

        ckpt_path = OUTPUT_DIR / f"music_transformer_ep{start_ep}.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        Q       = ckpt['Q']
        D_EMBED = ckpt['d_embed']
        D_MODEL = ckpt['d_model']
        SEQ_LEN = ckpt['seq_len']
        N_LAYER = ckpt['n_layer']
        N_HEAD  = ckpt['n_head']

        model = CodebookMultiHeadTransformer(VOCAB_SIZE, D_EMBED, Q, D_MODEL, SEQ_LEN, N_LAYER, N_HEAD).to(DEVICE)
        model.load_state_dict(strip_prefixes(ckpt['model_state_dict']), strict=True)
        try:
            model = torch.compile(model)
        except Exception:
            pass

        ds, dl = make_loader(META_PATH, SEQ_LEN, Q, BATCH_SIZE, DEVICE)
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
            'Q': Q, 'd_embed': D_EMBED, 'd_model': D_MODEL, 'seq_len': SEQ_LEN,
            'n_layer': N_LAYER, 'n_head': N_HEAD, 'vocab_size': VOCAB_SIZE
        }
        train_loop(model, dl, opt, scheduler, scaler, DEVICE, ACCUM_STEPS, MAX_NORM, start_ep, target_ep, OUTPUT_DIR, meta)

    else:
        raise SystemExit("Please type 'new' or 'resume'.")
