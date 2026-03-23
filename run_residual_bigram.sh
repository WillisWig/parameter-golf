#!/bin/bash

set -euo pipefail

MODE=${1:-smoke}
NGPU=${2:-1}

if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

case "$MODE" in
  smoke)
    RUN_ID=residual_bigram_smoke \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    ITERATIONS=200 \
    TRAIN_BATCH_TOKENS=8192 \
    TRAIN_LOG_EVERY=20 \
    VAL_LOSS_EVERY=0 \
    VAL_BATCH_SIZE=524288 \
    USE_BIGRAM_PRIOR=1 \
    BIGRAM_PRIOR_BUCKETS=10240 \
    BIGRAM_PRIOR_DIM=128 \
    BIGRAM_PRIOR_GATE_INIT=0.05 \
    python3 train_gpt_residual_bigram.py
    ;;
  full)
    if [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000079.bin" ]; then
      python3 data/cached_challenge_fineweb.py --variant sp1024
    fi
    NCCL_IB_DISABLE=1 \
    RUN_ID=residual_bigram_full \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    MAX_WALLCLOCK_SECONDS=600 \
    TRAIN_BATCH_TOKENS=524288 \
    TRAIN_LOG_EVERY=50 \
    VAL_LOSS_EVERY=200 \
    VAL_BATCH_SIZE=524288 \
    USE_BIGRAM_PRIOR=1 \
    BIGRAM_PRIOR_BUCKETS=10240 \
    BIGRAM_PRIOR_DIM=128 \
    BIGRAM_PRIOR_GATE_INIT=0.05 \
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt_residual_bigram.py
    ;;
  *)
    echo "Usage: $0 [smoke|full] [num_gpus]"
    exit 1
    ;;
esac
