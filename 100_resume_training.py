# 100_resume_training.py ─ Resume training from an existing checkpoint
import json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Config, GPT2Model
from tqdm.auto import tqdm

# ─────────── Specials (must match the code you trained with) ───────────
BOS_ID  = 1024          # first special token, one larger than any Encodec code
PAD_ID  = 1025          # padding token
VOCAB_SIZE = PAD_ID + 1 # 1026 rows in every embedding table
IGNORE_INDEX = -100     # loss is not computed on these positions

# ─────────── Fixed hyper-params (tweak if you like) ───────────
folder_name   = input("Enter the name of artist: ")
OUTPUT_DIR    = Path(f"model_results_{folder_name}")
META_PATH     = Path(f"metadata_clean_{folder_name}.jsonl")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 4                   # keep this low if you’re RAM-bound
LR            = float(input("Enter the learning rate (e.g. 2e-4): "))
TARGET_EPOCH  = 600                 # stop when we reach this epoch

# ─────────── Dataset ───────────
class EncodecTokenDataset(Dataset):
    def __init__(self, meta_path: Path, seq_len: int):
        with open(meta_path) as f:
            self.rows = [json.loads(l) for l in f]
        self.seq_len = seq_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        tpath = Path(row["audio"])
        tokens = torch.load(tpath)[0].transpose(0, 1)           # [T, Q]
        Q = tokens.shape[1]

        # prepend BOS
        bos = torch.full((1, Q), BOS_ID, dtype=torch.long)
        tokens = torch.cat([bos, tokens], 0)

        # pad / trim to (seq_len + 1)
        if tokens.shape[0] < self.seq_len + 1:
            pad_len = self.seq_len + 1 - tokens.shape[0]
            pad = torch.full((pad_len, Q), PAD_ID, dtype=torch.long)
            tokens = torch.cat([tokens, pad], 0)
        else:
            tokens = tokens[: self.seq_len + 1]

        input_ids = tokens[:-1]              # may contain PAD_ID
        labels    = tokens[1:].clone()
        labels[labels == PAD_ID] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}

# ─────────── Model ───────────
class CodebookMultiHeadTransformer(nn.Module):
    def __init__(self, vocab_size, d_embed, Q, d_model, seq_len, n_layer, n_head):
        super().__init__()
        self.Q = Q
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, d_embed) for _ in range(Q)]
        )
        self.linear_proj = nn.Linear(Q * d_embed, d_model)
        cfg = GPT2Config(
            vocab_size=1, 
            n_positions=seq_len,
            n_embd=d_model, 
            n_layer=n_layer, 
            n_head=n_head,
            resid_pdrop = 0.15,       # dropout on MLP outputs
            embd_pdrop = 0.05,       # dropout on token/pos embeds
            attn_pdrop = 0.05        # dropout on attention weights
            )
        self.transformer = GPT2Model(cfg)
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(Q)])

    def forward(self, input_ids, labels=None):
        B, T, Q = input_ids.shape
        embeds  = [self.embeddings[q](input_ids[:,:,q]) for q in range(Q)]
        hidden  = self.transformer(
            inputs_embeds=self.linear_proj(torch.cat(embeds, -1))
        ).last_hidden_state                                           # [B,T,D]
        if labels is None:                                           # generation
            return self.heads, hidden
        loss = 0
        for q in range(Q):
            logits_q = self.heads[q](hidden)
            loss += F.cross_entropy(
                logits_q.view(-1, VOCAB_SIZE),
                labels[:,:,q].reshape(-1),
                ignore_index=IGNORE_INDEX,
                label_smoothing=0.1
            )
        return loss

# ─────────── Training loop to resume ───────────
def resume_training_from_checkpoint(start_epoch: int):
    ckpt = torch.load(OUTPUT_DIR / f"music_transformer_ep{start_epoch}.pt",
                      map_location=DEVICE)

    model = CodebookMultiHeadTransformer(
        vocab_size = ckpt.get("vocab_size", VOCAB_SIZE),
        d_embed    = ckpt["d_embed"],
        Q          = ckpt["Q"],
        d_model    = ckpt["d_model"],
        seq_len    = ckpt["seq_len"],
        n_layer    = ckpt["n_layer"],
        n_head     = ckpt["n_head"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    opt = AdamW(model.parameters(), lr=LR)
    opt.load_state_dict(ckpt["optimizer_state_dict"])


    dl  = DataLoader(
        EncodecTokenDataset(META_PATH, ckpt["seq_len"]),
        batch_size=BATCH_SIZE, shuffle=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dl) * (TARGET_EPOCH - start_epoch))
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])  # ✅ resume scheduler state

    print(f"🚀 Resuming at epoch {ckpt['epoch']} → target {TARGET_EPOCH}")
    for ep in range(ckpt["epoch"] + 1, TARGET_EPOCH + 1):
        model.train(); running_loss = 0; steps = 0
        for batch in tqdm(dl):
            inp = batch["input_ids"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            loss = model(inp, labels=lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            running_loss += loss.item(); steps += 1
        print(f"✅  Epoch {ep} — Avg loss {running_loss/steps:.4f}")

        #if ep % 10 == 0:
        torch.save({**ckpt,        # keep original meta
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": running_loss/steps},
                    OUTPUT_DIR / f"music_transformer_ep{ep}.pt")
        print(f"📦 Saved checkpoint to {OUTPUT_DIR / f'music_transformer_ep{ep}.pt'}")

    print("🎉  Finished training up to", TARGET_EPOCH)

# ─────────── Main ───────────
if __name__ == "__main__":
    start_ep = int(input("🔁  Resume from epoch: "))
    resume_training_from_checkpoint(start_ep)
