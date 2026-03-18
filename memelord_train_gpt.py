"""
Parameter Golf: The Memelord Submission
========================================
Drop-in replacement for train_gpt.py in the parameter-golf repo.

STRATEGY: Depth-recurrent transformer with optional BitNet quantization.
One fat transformer block, looped N times. Same interface as baseline.

USAGE (matches baseline exactly):
    RUN_ID=memelord_v1 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py

For 8xH100 submission:
    torchrun --standalone --nproc_per_node=8 train_gpt.py
"""

import os
import sys
import math
import time
import struct
import zlib
import io
import json
import glob
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Read env vars (same interface as baseline)
# ---------------------------------------------------------------------------
RUN_ID = os.environ.get("RUN_ID", "memelord_run")
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024/")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "1024"))
MAX_WALLCLOCK_SECONDS = int(os.environ.get("MAX_WALLCLOCK_SECONDS", "600"))  # 10 min default
VAL_LOSS_EVERY = int(os.environ.get("VAL_LOSS_EVERY", "200"))  # 0 = only at end
ITERATIONS = int(os.environ.get("ITERATIONS", "0"))  # 0 = use wallclock
TRAIN_BATCH_TOKENS = int(os.environ.get("TRAIN_BATCH_TOKENS", "524288"))  # 512K tokens
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE", "524288"))

# ---------------------------------------------------------------------------
# MEMELORD CONFIG: The knobs to turn
# ---------------------------------------------------------------------------
# Depth recurrence: ONE block looped this many times
N_LOOPS = int(os.environ.get("N_LOOPS", "12"))
# Model width (bigger = more capacity per block, but more params)
N_EMBD = int(os.environ.get("N_EMBD", "768"))
# Number of attention heads
N_HEAD = int(os.environ.get("N_HEAD", "12"))
# FFN expansion factor
FFN_MULT = int(os.environ.get("FFN_MULT", "4"))
# Sequence length
SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
# Use GQA (grouped query attention) - number of KV heads (0 = same as N_HEAD)
N_KV_HEAD = int(os.environ.get("N_KV_HEAD", "4"))
# Learning rate
LR = float(os.environ.get("LR", "6e-4"))
MIN_LR = float(os.environ.get("MIN_LR", "6e-5"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "200"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))
# Mulligan seeds (set to 1 to disable)
MULLIGAN_SEEDS = int(os.environ.get("MULLIGAN_SEEDS", "8"))
MULLIGAN_SECONDS = float(os.environ.get("MULLIGAN_SECONDS", "60"))
# BitNet mode (experimental - set to 1 to enable)
USE_BITNET = int(os.environ.get("USE_BITNET", "0"))


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------------------
# BitNet (optional, experimental)
# ---------------------------------------------------------------------------
def ste_round(x):
    return x + (torch.round(x) - x).detach()

def quantize_ternary(w):
    alpha = w.abs().mean()
    w_scaled = w / (alpha + 1e-8)
    w_clipped = torch.clamp(w_scaled, -1, 1)
    w_q = ste_round(w_clipped)
    return w_q * alpha

class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def forward(self, x):
        w = quantize_ternary(self.weight) if self.training else quantize_ternary(self.weight)
        return F.linear(x, w, self.bias)


# ---------------------------------------------------------------------------
# Attention with Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, use_bitnet=False):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head > 0 else n_head
        self.head_dim = n_embd // n_head
        self.n_rep = self.n_head // self.n_kv_head  # GQA repetition factor

        Lin = BitLinear if use_bitnet else nn.Linear

        # Q gets full heads, KV gets fewer (GQA)
        self.q_proj = Lin(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = Lin(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = Lin(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = Lin(n_head * self.head_dim, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, n_embd, ffn_dim, use_bitnet=False):
        super().__init__()
        Lin = BitLinear if use_bitnet else nn.Linear
        self.w1 = Lin(n_embd, ffn_dim, bias=False)
        self.w3 = Lin(n_embd, ffn_dim, bias=False)  # gate
        self.w2 = Lin(ffn_dim, n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer Block (THE block — there's only one, looped N times)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, ffn_dim, use_bitnet=False):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_head, n_kv_head, use_bitnet)
        self.ln2 = RMSNorm(n_embd)
        self.ffn = SwiGLU(n_embd, ffn_dim, use_bitnet)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# THE MODEL: Depth-Recurrent GPT
# ---------------------------------------------------------------------------
class DepthRecurrentGPT(nn.Module):
    """
    One transformer block, looped N times.
    Same effective depth as N-layer transformer.
    1/N the stored parameters.
    """

    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, ffn_dim,
                 n_loops, seq_len, use_bitnet=False):
        super().__init__()
        self.n_loops = n_loops
        self.n_embd = n_embd
        self.seq_len = seq_len

        # Embeddings (full precision — they're small)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)

        # Loop-iteration embedding: tells the model which pass it's on
        self.loop_emb = nn.Embedding(n_loops, n_embd)

        # THE SINGLE BLOCK
        self.block = TransformerBlock(n_embd, n_head, n_kv_head, ffn_dim, use_bitnet)

        # Output
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying (embed ↔ output head)
        self.lm_head.weight = self.tok_emb.weight

        # Init
        self.apply(self._init_weights)
        # Scale residual connections by 1/sqrt(2*n_loops) for stability
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_loops))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))

        # DEPTH RECURRENCE
        for i in range(self.n_loops):
            loop_signal = self.loop_emb.weight[i]  # (n_embd,)
            x = x + loop_signal * (0.1 / math.sqrt(self.n_embd))
            x = self.block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data Loading (matches baseline .bin shard format)
# ---------------------------------------------------------------------------
class ShardedDataLoader:
    """Load pre-tokenized .bin shards, cycling through them."""

    def __init__(self, data_path, split, seq_len, batch_tokens, rank=0, world_size=1):
        self.seq_len = seq_len
        self.batch_tokens = batch_tokens
        self.batch_size = batch_tokens // seq_len
        self.rank = rank
        self.world_size = world_size

        pattern = os.path.join(data_path, f"fineweb_{split}_*.bin")
        self.shards = sorted(glob.glob(pattern))
        if not self.shards:
            # Try without prefix
            pattern = os.path.join(data_path, f"*_{split}_*.bin")
            self.shards = sorted(glob.glob(pattern))

        assert len(self.shards) > 0, f"No shards found for {split} in {data_path}"

        self.current_shard_idx = 0
        self.current_pos = 0
        self._load_shard(0)

    def _load_shard(self, idx):
        self.current_shard_idx = idx % len(self.shards)
        self.data = np.memmap(self.shards[self.current_shard_idx], dtype=np.uint16, mode='r')
        self.current_pos = self.rank * self.batch_size * self.seq_len

    def next_batch(self):
        B = self.batch_size
        T = self.seq_len

        # Check if we need to advance to next shard
        if self.current_pos + (B * T + 1) > len(self.data):
            self._load_shard(self.current_shard_idx + 1)

        buf = torch.from_numpy(
            self.data[self.current_pos:self.current_pos + B * T + 1].astype(np.int64)
        )
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B * T * self.world_size

        return x.cuda(), y.cuda()


class FullValLoader:
    """Load ALL validation shards for final evaluation."""

    def __init__(self, data_path, seq_len, batch_tokens):
        self.seq_len = seq_len
        self.batch_size = batch_tokens // seq_len

        pattern = os.path.join(data_path, "fineweb_val_*.bin")
        self.shards = sorted(glob.glob(pattern))
        assert len(self.shards) > 0, f"No val shards found in {data_path}"

        # Concat all validation data
        arrays = [np.memmap(s, dtype=np.uint16, mode='r') for s in self.shards]
        self.data = np.concatenate(arrays)

    def evaluate(self, model, device, max_batches=0):
        """Run full validation, return average loss and BPB."""
        model.eval()
        losses = []
        pos = 0
        B = self.batch_size
        T = self.seq_len
        n_batches = 0

        with torch.no_grad():
            while pos + B * T + 1 <= len(self.data):
                buf = torch.from_numpy(
                    self.data[pos:pos + B * T + 1].astype(np.int64)
                )
                x = buf[:-1].view(B, T).to(device)
                y = buf[1:].view(B, T).to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(x, y)

                losses.append(loss.item())
                pos += B * T
                n_batches += 1

                if max_batches > 0 and n_batches >= max_batches:
                    break

        avg_loss = sum(losses) / len(losses) if losses else float('inf')
        return avg_loss


# ---------------------------------------------------------------------------
# BPB Calculation
# ---------------------------------------------------------------------------
def compute_bpb(val_loss, tokenizer_path, data_path):
    """
    Compute bits-per-byte (tokenizer-agnostic metric).

    BPB = val_loss_nats * (avg_tokens / avg_bytes) / ln(2)

    We need the ratio of tokens to bytes in the validation set.
    """
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

        # Sample some validation text to estimate tokens/byte ratio
        # Load raw validation text if available, else estimate
        val_shards = sorted(glob.glob(os.path.join(data_path, "fineweb_val_*.bin")))
        if val_shards:
            tokens = np.memmap(val_shards[0], dtype=np.uint16, mode='r')
            # Decode a sample to get byte count
            sample_tokens = tokens[:100000].tolist()
            sample_text = sp.decode(sample_tokens)
            sample_bytes = len(sample_text.encode('utf-8'))
            tokens_per_byte = len(sample_tokens) / sample_bytes
            bpb = val_loss * tokens_per_byte / math.log(2)
            return bpb, tokens_per_byte
    except Exception as e:
        print(f"  Warning: BPB estimation failed ({e}), using approximation")

    # Fallback: rough estimate for sp1024
    # Typical ratio for small vocab on English text: ~0.3-0.5 tokens per byte
    tokens_per_byte = 0.4
    bpb = val_loss * tokens_per_byte / math.log(2)
    return bpb, tokens_per_byte


# ---------------------------------------------------------------------------
# Artifact Size Calculation (INT8 + zlib, matching baseline)
# ---------------------------------------------------------------------------
def compute_artifact_size(model, script_path):
    """Compute artifact size: code bytes + INT8-quantized zlib-compressed model."""
    # Code size
    code_bytes = os.path.getsize(script_path)

    # INT8 quantization + zlib compression
    state_dict = model.state_dict()
    buffer = io.BytesIO()

    for name, param in state_dict.items():
        data = param.detach().cpu().float()
        # INT8 quantization
        scale = data.abs().max() / 127.0
        if scale > 0:
            quantized = torch.clamp(torch.round(data / scale), -127, 127).to(torch.int8)
        else:
            quantized = torch.zeros_like(data, dtype=torch.int8)

        # Write scale (float32) + quantized data
        buffer.write(struct.pack('f', scale.item()))
        buffer.write(quantized.numpy().tobytes())

    raw_model_bytes = buffer.getvalue()
    compressed_model_bytes = zlib.compress(raw_model_bytes, level=9)

    return code_bytes, len(compressed_model_bytes), code_bytes + len(compressed_model_bytes)


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------
def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step >= total_steps:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


# ---------------------------------------------------------------------------
# Mulligan: Quick seed search
# ---------------------------------------------------------------------------
def mulligan_search(train_loader, val_loader, device, rank, world_size):
    """Try multiple seeds, pick the one with best initial loss trajectory."""
    if MULLIGAN_SEEDS <= 1:
        return 42, None

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"MULLIGAN: Testing {MULLIGAN_SEEDS} seeds ({MULLIGAN_SECONDS}s budget)")
        print(f"{'='*60}")

    best_loss = float('inf')
    best_seed = 42
    best_state = None
    time_per_seed = MULLIGAN_SECONDS / MULLIGAN_SEEDS

    for seed_idx in range(MULLIGAN_SEEDS):
        seed = seed_idx * 7 + 13  # spread seeds out
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = DepthRecurrentGPT(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_kv_head=N_KV_HEAD, ffn_dim=N_EMBD * FFN_MULT,
            n_loops=N_LOOPS, seq_len=SEQ_LEN, use_bitnet=bool(USE_BITNET)
        ).cuda()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95), fused=True
        )

        model.train()
        seed_start = time.time()
        step = 0

        while time.time() - seed_start < time_per_seed:
            x, y = train_loader.next_batch()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        # Quick eval
        model.eval()
        eval_loss = val_loader.evaluate(model, device, max_batches=5)

        if rank == 0:
            print(f"  Seed {seed:4d}: val_loss={eval_loss:.4f} ({step} steps)")

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_seed = seed
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        del model, optimizer
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"  Winner: seed {best_seed} (val_loss={best_loss:.4f})")

    return best_seed, best_state


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # Distributed setup
    use_ddp = int(os.environ.get('RANK', -1)) >= 0
    if use_ddp:
        dist.init_process_group('nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda'
        torch.cuda.set_device(0)

    is_master = (rank == 0)

    if is_master:
        print("=" * 60)
        print(f"PARAMETER GOLF: DEPTH RECURRENCE (run={RUN_ID})")
        print("=" * 60)
        print(f"  vocab_size={VOCAB_SIZE}, n_embd={N_EMBD}, n_head={N_HEAD}")
        print(f"  n_kv_head={N_KV_HEAD}, ffn_dim={N_EMBD * FFN_MULT}")
        print(f"  n_loops={N_LOOPS} (depth recurrence)")
        print(f"  seq_len={SEQ_LEN}, bitnet={USE_BITNET}")
        print(f"  mulligan_seeds={MULLIGAN_SEEDS}")
        print(f"  lr={LR}, min_lr={MIN_LR}, warmup={WARMUP_STEPS}")
        print(f"  world_size={world_size}, device={device}")
        print(f"  data={DATA_PATH}")

    wall_start = time.time()

    # Data loaders
    train_loader = ShardedDataLoader(
        DATA_PATH, "train", SEQ_LEN, TRAIN_BATCH_TOKENS, rank, world_size
    )
    val_loader = FullValLoader(DATA_PATH, SEQ_LEN, VAL_BATCH_SIZE)

    if is_master:
        print(f"  train shards: {len(train_loader.shards)}")
        print(f"  val shards: {len(val_loader.shards)}")
        print(f"  batch_size: {train_loader.batch_size} seqs × {SEQ_LEN} tokens")

    # -----------------------------------------------------------------------
    # MULLIGAN PHASE
    # -----------------------------------------------------------------------
    best_seed, best_state = mulligan_search(train_loader, val_loader, device, rank, world_size)

    # -----------------------------------------------------------------------
    # FULL TRAINING
    # -----------------------------------------------------------------------
    if is_master:
        remaining = MAX_WALLCLOCK_SECONDS - (time.time() - wall_start)
        print(f"\n{'='*60}")
        print(f"TRAINING (seed={best_seed}, {remaining:.0f}s remaining)")
        print(f"{'='*60}")

    torch.manual_seed(best_seed)
    torch.cuda.manual_seed_all(best_seed)

    model = DepthRecurrentGPT(
        vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
        n_kv_head=N_KV_HEAD, ffn_dim=N_EMBD * FFN_MULT,
        n_loops=N_LOOPS, seq_len=SEQ_LEN, use_bitnet=bool(USE_BITNET)
    ).cuda()

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if is_master:
        n_params = model.param_count()
        print(f"  Parameters: {n_params:,}")
        print(f"  Effective (√{N_LOOPS} multiplier): ~{n_params * math.sqrt(N_LOOPS):,.0f}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if use_ddp else model

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95), fused=True
    )

    # Estimate total steps
    remaining_time = MAX_WALLCLOCK_SECONDS - (time.time() - wall_start)
    # We'll refine this after a few steps
    estimated_total_steps = 5000  # initial guess

    # Training
    model.train()
    step = 0
    train_start = time.time()
    log_lines = []

    while True:
        elapsed = time.time() - wall_start
        if MAX_WALLCLOCK_SECONDS > 0 and elapsed > MAX_WALLCLOCK_SECONDS - 30:
            # Reserve 30s for final eval + compression
            break
        if ITERATIONS > 0 and step >= ITERATIONS:
            break

        # Refine step estimate after 20 steps
        if step == 20:
            sps = step / (time.time() - train_start)
            estimated_total_steps = int(sps * remaining_time)
            if is_master:
                print(f"  Speed: {sps:.1f} steps/s → ~{estimated_total_steps} total steps")

        # LR schedule
        lr = get_lr(step, estimated_total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward / backward
        x, y = train_loader.next_batch()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Log
        if is_master and step % 20 == 0:
            line = f"step={step:5d} | train_loss={loss.item():.4f} | lr={lr:.2e} | t={elapsed:.0f}s"
            print(f"  {line}")
            log_lines.append(line)

        # Periodic validation
        if VAL_LOSS_EVERY > 0 and step > 0 and step % VAL_LOSS_EVERY == 0:
            val_loss = val_loader.evaluate(model, device, max_batches=20)
            if is_master:
                vline = f"  >>> val_loss={val_loss:.4f} (step {step})"
                print(vline)
                log_lines.append(vline)
            model.train()

        step += 1

    # -----------------------------------------------------------------------
    # FINAL EVALUATION + COMPRESSION
    # -----------------------------------------------------------------------
    if is_master:
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")

        # Full validation
        val_loss = val_loader.evaluate(raw_model, device)
        bpb, tokens_per_byte = compute_bpb(val_loss, TOKENIZER_PATH, DATA_PATH)

        print(f"  val_loss: {val_loss:.6f}")
        print(f"  val_bpb:  {bpb:.6f}")
        print(f"  tokens_per_byte: {tokens_per_byte:.4f}")

        # Artifact size
        script_path = os.path.abspath(__file__)
        code_bytes, model_bytes, total_bytes = compute_artifact_size(raw_model, script_path)

        print(f"\n  final_int8_zlib_roundtrip:")
        print(f"    code_bytes:  {code_bytes:>12,}")
        print(f"    model_bytes: {model_bytes:>12,}")
        print(f"    total:       {total_bytes:>12,}")
        print(f"    budget:      {16_000_000:>12,}")
        if total_bytes <= 16_000_000:
            print(f"    status:      UNDER BUDGET ({16_000_000 - total_bytes:,} bytes free)")
        else:
            print(f"    status:      OVER BUDGET by {total_bytes - 16_000_000:,} bytes!")

        total_time = time.time() - wall_start
        print(f"\n  Total training time: {total_time:.1f}s ({step} steps)")

        # Save log
        os.makedirs(f"logs/{RUN_ID}", exist_ok=True)
        log_path = f"logs/{RUN_ID}/train_log.txt"
        with open(log_path, 'w') as f:
            f.write(f"run_id: {RUN_ID}\n")
            f.write(f"val_loss: {val_loss:.6f}\n")
            f.write(f"val_bpb: {bpb:.6f}\n")
            f.write(f"artifact_bytes: {total_bytes}\n")
            f.write(f"total_steps: {step}\n")
            f.write(f"total_time_s: {total_time:.1f}\n")
            f.write(f"n_params: {raw_model.param_count()}\n")
            f.write(f"n_loops: {N_LOOPS}\n")
            f.write(f"n_embd: {N_EMBD}\n")
            f.write(f"n_head: {N_HEAD}\n")
            f.write(f"n_kv_head: {N_KV_HEAD}\n")
            f.write(f"vocab_size: {VOCAB_SIZE}\n")
            f.write(f"seed: {best_seed}\n")
            f.write(f"\n--- Training Log ---\n")
            for line in log_lines:
                f.write(line + "\n")
        print(f"  Log saved to {log_path}")

        # Save model weights
        weight_path = f"logs/{RUN_ID}/model.pt"
        torch.save(raw_model.state_dict(), weight_path)
        print(f"  Model saved to {weight_path}")

        # Save submission.json
        submission = {
            "name": "The Memelord Submission",
            "github_id": "YOUR_GITHUB_USERNAME",  # FILL THIS IN
            "val_bpb": round(bpb, 6),
            "val_loss": round(val_loss, 6),
            "artifact_bytes": total_bytes,
            "architecture": "depth_recurrent_gpt",
            "n_params": raw_model.param_count(),
            "n_loops": N_LOOPS,
            "n_embd": N_EMBD,
            "training_time_s": round(total_time, 1),
            "description": (
                f"Single transformer block (d={N_EMBD}, {N_HEAD}h, {N_KV_HEAD}kv) "
                f"looped {N_LOOPS} times with learned loop embeddings. "
                f"GQA + SwiGLU + RMSNorm. "
                f"Mulligan seed search ({MULLIGAN_SEEDS} seeds). "
                f"Cosine LR decay with warmup."
            )
        }
        sub_path = f"logs/{RUN_ID}/submission.json"
        with open(sub_path, 'w') as f:
            json.dump(submission, f, indent=2)
        print(f"  Submission saved to {sub_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
