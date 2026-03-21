#!/bin/bash

set -euo pipefail

MODE=${1:-smoke}
NGPU=${2:-1}

if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
  if [ ! -f "./data/cached_challenge_fineweb.py" ]; then
    echo "Missing ./data/cached_challenge_fineweb.py. Run this inside a full parameter-golf checkout."
    exit 1
  fi
  if [ "$MODE" = "smoke" ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
  else
    python3 data/cached_challenge_fineweb.py --variant sp1024
  fi
fi

case "$MODE" in
  smoke)
    RUN_ID=middleout_smoke \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    ITERATIONS=200 \
    TRAIN_BATCH_TOKENS=8192 \
    TRAIN_LOG_EVERY=20 \
    VAL_LOSS_EVERY=0 \
    VAL_BATCH_SIZE=524288 \
    DEPTH_RECURRENCE=1 \
    NUM_LAYERS=12 \
    MODEL_DIM=768 \
    NUM_HEADS=12 \
    NUM_KV_HEADS=4 \
    MLP_MULT=2 \
    PREV_TOKEN_SMEAR=1 \
    BIGRAM_HASH_BUCKETS=4096 \
    TRIGRAM_HASH_BUCKETS=8192 \
    NGRAM_HASH_DIM=64 \
    python3 train_gpt_recurrent.py
    ;;
  full)
    NCCL_IB_DISABLE=1 \
    RUN_ID=middleout_full \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    MAX_WALLCLOCK_SECONDS=600 \
    TRAIN_BATCH_TOKENS=524288 \
    TRAIN_LOG_EVERY=50 \
    VAL_LOSS_EVERY=200 \
    VAL_BATCH_SIZE=524288 \
    DEPTH_RECURRENCE=1 \
    NUM_LAYERS=12 \
    MODEL_DIM=768 \
    NUM_HEADS=12 \
    NUM_KV_HEADS=4 \
    MLP_MULT=2 \
    PREV_TOKEN_SMEAR=1 \
    BIGRAM_HASH_BUCKETS=4096 \
    TRIGRAM_HASH_BUCKETS=8192 \
    NGRAM_HASH_DIM=64 \
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt_recurrent.py
    ;;
  *)
    echo "Usage: $0 [smoke|full] [num_gpus]"
    exit 1
    ;;
esac
