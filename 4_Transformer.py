# 4_Transformer.py — Train GPT2-style model on Q-codebook tokens with separate heads

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Config, GPT2Model
from tqdm.auto import tqdm

# ─────────── CONFIG ───────────
folder_name = input("Enter the name of artist: ")

OUTPUT_DIR = Path(f"model_results_{folder_name}")
OUTPUT_DIR.mkdir(exist_ok=True)

TOKEN_ROOT  = Path(f"encoded_tokens/{folder_name}")
META_PATH   = Path(f"metadata_clean_{folder_name}.jsonl")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────── Specials ───────────
BOS_ID  = 1024          # first id after the 0-1023 Encodec codes
PAD_ID  = 1025          # one step higher
VOCAB_SIZE = PAD_ID + 1 # 1026 rows in every embedding table

# ─────────── Infer SEQ_LEN ───────────
def infer_seq_len(encoded_dir: str = "encoded_tokens") -> int:
    encoded_path = Path(encoded_dir)
    token_files  = list(encoded_path.rglob("*.th"))
    if not token_files:
        raise FileNotFoundError(f"No .th token files found in {encoded_dir}")

    sample = torch.load(token_files[0])
    if sample.ndim != 3:
        raise ValueError(f"Expected token shape [1, Q, T], got {sample.shape}")
    _, Q, T = sample.shape
    print(f"🔍 Inferred SEQ_LEN = {T} from {token_files[0].name}")
    return T, Q

SEQ_LEN, Q = infer_seq_len(f"encoded_tokens/{folder_name}")
BATCH_SIZE  = 5          # how many samples per gradient step
LR          = 5e-4       # how much to adjust after each back-propagation
EPOCHS      = 500        # Number of full passes over the entire training dataset
D_EMBED     = 64            # Each token in a codebook is embedded into a 64-dimensional vector
D_MODEL     = Q * D_EMBED # Final input dimension to the transformer
N_LAYER     = 8  # Number of transformer decoder layers (blocks)
N_HEAD      = 8  # Number of attention heads in the multi-head self-attention mechanism

# ─────────── Dataset ───────────
class EncodecTokenDataset(Dataset):
    def __init__(self, meta_path: Path, seq_len: int):
        with open(meta_path) as f:
            self.rows = [json.loads(l) for l in f]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        tpath = Path(row["audio"])
        tokens = torch.load(tpath)[0].transpose(0, 1)   # [T, Q]

        # prepend BOS
        bos = torch.full((1, Q), BOS_ID, dtype=torch.long)
        tokens = torch.cat([bos, tokens], 0)            # [T+1, Q]

        # pad / trim
        if tokens.shape[0] < self.seq_len + 1:
            pad_len = self.seq_len + 1 - tokens.shape[0]
            pad = torch.full((pad_len, Q), PAD_ID, dtype=torch.long)
            tokens = torch.cat([tokens, pad], 0)
        else:
            tokens = tokens[: self.seq_len + 1]

        # split into inputs / labels
        input_ids = tokens[:-1]          # may contain PAD_ID, never -100
        labels    = tokens[1:].clone()
        labels[labels == PAD_ID] = -100  # ignore pads in loss

        return {"input_ids": input_ids, "labels": labels}

    
# ─────────── Model ───────────
class CodebookMultiHeadTransformer(nn.Module):
    def __init__(self, vocab_size, d_embed, Q, d_model, seq_len, n_layer, n_head):
        super().__init__()
        self.Q = Q # Store Q

        # Embedding each input layer(for example, in this case, embed 1024 dimensions to 64 dimensions for all notebooks)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_embed) for _ in range(Q) # Returns 4(=rangeQ) embeddings (64, 64, 64, 64)
        ])
        self.linear_proj = nn.Linear(Q * d_embed, d_model) # Concatenate embeddings to a 1d vector, matching the dimension expected by the model

        # configure and create a GPT-2 Transformer
        config = GPT2Config(
            vocab_size=1,  # dummy
            n_positions=seq_len, # Max sequence length the model can handle
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop = 0.15,       # dropout on MLP outputs
            embd_pdrop = 0.10,       # dropout on token/pos embeds
            attn_pdrop = 0.10        # dropout on attention weights
        )
        self.transformer = GPT2Model(config)

        # Output prediction heads(Q heads in total)
        # Each head turns the Transformer output into predictions for one codebook
        # input: vector in shape of d_moel, output: vector in shape of vocab_size(which is a probability distribution)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(Q) 
        ])

    def forward(self, input_ids, labels=None):
        # input_ids: [B, T, Q]
        B, T, Q = input_ids.shape

        # loop through each codebook q
        # Extract the token stream from input (input_ids[:, :, q])
        # Get the embedding → shape [B, T, d_embed]
        embeds = [self.embeddings[q](input_ids[:, :, q]) for q in range(Q)]  # List of [B, T, d_embed]

        # concatenate along the last axis (feature dimension), making a new tensor of shape
        concat = torch.cat(embeds, dim=-1)  # [B, T, Q * d_embed]

        # Projects the concatenated vector down (or up) to the expected size for the Transformer
        inputs_embeds = self.linear_proj(concat)  # [B, T, d_model]

        # Returns the hidden states (contextualized vectors for each time step)
        hidden_states = self.transformer(inputs_embeds=inputs_embeds).last_hidden_state  # [B, T, d_model]

        # If generating (no labels), return outputs
        if labels is None:
            return self.heads, hidden_states

        # If training, calculate loss
        total_loss = 0
        for q in range(Q):
            logits_q = self.heads[q](hidden_states)  # [B, T, vocab_size], where the third dimension means the probability over all number of possible tokens

            # Compares the predicted logits and the true labels
            loss_q = F.cross_entropy(
                logits_q.view(-1, VOCAB_SIZE), # flattens it into [B × T, vocab_size]
                labels[:, :, q].reshape(-1), # grabs the true token labels for codebook q, shape before: [B, T], after: [B × T]
                ignore_index=-100 # Ignores positions where the label is 0
            )
            total_loss += loss_q

        return total_loss

# ─────────── Initialize ───────────
model = CodebookMultiHeadTransformer(
    vocab_size=VOCAB_SIZE,
    d_embed=D_EMBED,
    Q=Q,
    d_model=D_MODEL,
    seq_len=SEQ_LEN,
    n_layer=N_LAYER,
    n_head=N_HEAD
).to(DEVICE)

# ─────────── Training ───────────
ds = EncodecTokenDataset(META_PATH, SEQ_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
opt = AdamW(model.parameters(), lr=LR)

print("\U0001f680  Starting training …")
for ep in range(1, EPOCHS + 1):
    model.train()
    total_steps = 0
    running_loss = 0.0

    for batch in tqdm(dl):
        inp  = batch["input_ids"].to(DEVICE)     # [B, T, Q]
        labs = batch["labels"].to(DEVICE)        # [B, T, Q]

        loss = model(inp, labels=labs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()
        total_steps += 1

    epoch_loss = running_loss / total_steps
    print(f"✅  Epoch {ep}/{EPOCHS} — Avg Loss: {epoch_loss:.4f}")

    #if ep % 10 == 0:
    ckpt_path = OUTPUT_DIR / f"music_transformer_ep{ep}.pt"
    torch.save({
        'epoch': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': epoch_loss,
        'Q': Q,
        'd_embed': D_EMBED,
        'd_model': D_MODEL,
        'seq_len': SEQ_LEN,
        'n_layer': N_LAYER,
        'n_head': N_HEAD,
        'vocab_size': VOCAB_SIZE
    }, ckpt_path)
    print(f"📦  Saved checkpoint to {ckpt_path.resolve()}")

print("🎉  Training complete.")


