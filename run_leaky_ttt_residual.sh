#!/bin/bash

set -euo pipefail

MODE=${1:-smoke}
NGPU=${2:-1}

if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

case "$MODE" in
  smoke)
    RUN_ID=leaky_ttt_residual_smoke \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    NUM_LAYERS=11 \
    BIGRAM_VOCAB_SIZE=1536 \
    XSA_LAST_N=4 \
    SWA_ENABLED=1 \
    SWA_EVERY=50 \
    ROPE_DIMS=16 \
    LN_SCALE=1 \
    VE_ENABLED=1 \
    VE_DIM=128 \
    VE_LAYERS=9,10 \
    MUON_WD=0.04 \
    ADAM_WD=0.04 \
    MATRIX_LR=0.025 \
    SCALAR_LR=0.025 \
    TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    WARMDOWN_ITERS=3500 \
    ITERATIONS=200 \
    TRAIN_BATCH_TOKENS=8192 \
    TRAIN_SEQ_LEN=1024 \
    EVAL_SEQ_LEN=1024 \
    VAL_BATCH_SIZE=524288 \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=20 \
    EVAL_STRIDE=0 \
    TTT_ENABLED=0 \
    RESIDUAL_BIGRAM_PRIOR=1 \
    RESIDUAL_BIGRAM_PRIOR_BUCKETS=128 \
    RESIDUAL_BIGRAM_PRIOR_DIM=16 \
    RESIDUAL_BIGRAM_PRIOR_GATE_INIT=0.05 \
    python3 train_gpt_leaky_ttt_residual.py
    ;;
  full)
    if [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000079.bin" ]; then
      python3 data/cached_challenge_fineweb.py --variant sp1024
    fi
    NCCL_IB_DISABLE=1 \
    RUN_ID=leaky_ttt_residual_full \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    NUM_LAYERS=11 \
    BIGRAM_VOCAB_SIZE=1536 \
    XSA_LAST_N=4 \
    SWA_ENABLED=1 \
    SWA_EVERY=50 \
    ROPE_DIMS=16 \
    LN_SCALE=1 \
    LATE_QAT_THRESHOLD=0.15 \
    VE_ENABLED=1 \
    VE_DIM=128 \
    VE_LAYERS=9,10 \
    TTT_ENABLED=1 \
    TTT_LR=0.002 \
    TTT_EPOCHS=3 \
    TTT_CHUNK_TOKENS=32768 \
    TTT_FREEZE_BLOCKS=0 \
    TTT_MOMENTUM=0.9 \
    TTT_BATCH_SEQS=32 \
    TTT_GRAD_CLIP=1.0 \
    MUON_WD=0.04 \
    ADAM_WD=0.04 \
    MATRIX_LR=0.025 \
    SCALAR_LR=0.025 \
    TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    WARMDOWN_ITERS=3500 \
    ITERATIONS=9000 \
    MAX_WALLCLOCK_SECONDS=600 \
    EVAL_STRIDE=64 \
    RESIDUAL_BIGRAM_PRIOR=1 \
    RESIDUAL_BIGRAM_PRIOR_BUCKETS=128 \
    RESIDUAL_BIGRAM_PRIOR_DIM=16 \
    RESIDUAL_BIGRAM_PRIOR_GATE_INIT=0.05 \
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt_leaky_ttt_residual.py
    ;;
  *)
    echo "Usage: $0 [smoke|full] [num_gpus]"
    exit 1
    ;;
esac
