#!/bin/bash
# ==============================================================================
# PARAMETER GOLF: The Memelord Quick Start
# ==============================================================================
# This script does everything. Run it on your RunPod machine.
#
# Usage:
#   chmod +x run_memelord.sh
#   ./run_memelord.sh          # Full run with depth recurrence
#   ./run_memelord.sh baseline # Run the original baseline first
#   ./run_memelord.sh smoke    # Quick smoke test (200 iterations)
#   ./run_memelord.sh sweep    # Test multiple configs
# ==============================================================================

set -e

MODE=${1:-"full"}
NGPU=${2:-1}  # Number of GPUs (1 for testing, 8 for submission)

echo "============================================================"
echo "  PARAMETER GOLF: THE MEMELORD ($MODE mode, ${NGPU}x GPU)"
echo "============================================================"

# --- Setup ---
cd /workspace
if [ ! -d "parameter-golf" ]; then
    echo "Cloning repo..."
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf

# --- Download data if needed ---
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading FineWeb data..."
    if [ "$MODE" = "smoke" ]; then
        python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
    else
        python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
    fi
fi

# --- Run based on mode ---
case $MODE in

    "baseline")
        echo ""
        echo "Running BASELINE (for comparison)..."
        echo ""
        RUN_ID=baseline_reference \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt.py
        ;;

    "smoke")
        echo ""
        echo "Running MEMELORD smoke test (200 iterations)..."
        echo ""
        # Copy our script
        cp memelord_train_gpt.py train_gpt_memelord.py 2>/dev/null || true

        RUN_ID=memelord_smoke \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        ITERATIONS=200 \
        MAX_WALLCLOCK_SECONDS=0 \
        VAL_LOSS_EVERY=0 \
        N_LOOPS=8 \
        N_EMBD=512 \
        N_HEAD=8 \
        N_KV_HEAD=4 \
        MULLIGAN_SEEDS=1 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt_memelord.py
        ;;

    "full")
        echo ""
        echo "Running FULL MEMELORD (10 min budget)..."
        echo ""
        cp memelord_train_gpt.py train_gpt_memelord.py 2>/dev/null || true

        RUN_ID=memelord_v1 \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        MAX_WALLCLOCK_SECONDS=600 \
        N_LOOPS=12 \
        N_EMBD=768 \
        N_HEAD=12 \
        N_KV_HEAD=4 \
        MULLIGAN_SEEDS=8 \
        MULLIGAN_SECONDS=60 \
        LR=6e-4 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt_memelord.py
        ;;

    "aggressive")
        echo ""
        echo "Running AGGRESSIVE config (wider model, more loops)..."
        echo ""
        cp memelord_train_gpt.py train_gpt_memelord.py 2>/dev/null || true

        RUN_ID=memelord_aggro \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        MAX_WALLCLOCK_SECONDS=600 \
        N_LOOPS=20 \
        N_EMBD=1024 \
        N_HEAD=16 \
        N_KV_HEAD=4 \
        MULLIGAN_SEEDS=8 \
        MULLIGAN_SECONDS=60 \
        LR=3e-4 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt_memelord.py
        ;;

    "bitnet")
        echo ""
        echo "Running BITNET config (experimental ternary weights)..."
        echo ""
        cp memelord_train_gpt.py train_gpt_memelord.py 2>/dev/null || true

        RUN_ID=memelord_bitnet \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        MAX_WALLCLOCK_SECONDS=600 \
        N_LOOPS=10 \
        N_EMBD=1024 \
        N_HEAD=16 \
        N_KV_HEAD=4 \
        MULLIGAN_SEEDS=4 \
        MULLIGAN_SECONDS=30 \
        USE_BITNET=1 \
        LR=3e-4 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt_memelord.py
        ;;

    "sweep")
        echo ""
        echo "Running CONFIG SWEEP (testing multiple setups)..."
        echo ""
        cp memelord_train_gpt.py train_gpt_memelord.py 2>/dev/null || true

        # Config 1: Conservative (similar to baseline but recurrent)
        echo "--- Config 1: Conservative (12 loops, 512d) ---"
        RUN_ID=sweep_conservative \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        ITERATIONS=500 \
        MAX_WALLCLOCK_SECONDS=0 \
        VAL_LOSS_EVERY=0 \
        N_LOOPS=12 \
        N_EMBD=512 \
        N_HEAD=8 \
        N_KV_HEAD=4 \
        MULLIGAN_SEEDS=1 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt_memelord.py

        # Config 2: Medium (wider)
        echo "--- Config 2: Medium (12 loops, 768d) ---"
        RUN_ID=sweep_medium \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        ITERATIONS=500 \
        MAX_WALLCLOCK_SECONDS=0 \
        VAL_LOSS_EVERY=0 \
        N_LOOPS=12 \
        N_EMBD=768 \
        N_HEAD=12 \
        N_KV_HEAD=4 \
        MULLIGAN_SEEDS=1 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt_memelord.py

        # Config 3: Aggressive (very wide, many loops)
        echo "--- Config 3: Aggressive (20 loops, 1024d) ---"
        RUN_ID=sweep_aggressive \
        DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
        TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
        VOCAB_SIZE=1024 \
        ITERATIONS=500 \
        MAX_WALLCLOCK_SECONDS=0 \
        VAL_LOSS_EVERY=0 \
        N_LOOPS=20 \
        N_EMBD=1024 \
        N_HEAD=16 \
        N_KV_HEAD=4 \
        MULLIGAN_SEEDS=1 \
        torchrun --standalone --nproc_per_node=$NGPU train_gpt_memelord.py

        echo ""
        echo "Sweep complete. Compare val_loss across:"
        echo "  logs/sweep_conservative/train_log.txt"
        echo "  logs/sweep_medium/train_log.txt"
        echo "  logs/sweep_aggressive/train_log.txt"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./run_memelord.sh [baseline|smoke|full|aggressive|bitnet|sweep] [num_gpus]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  DONE. Check logs/ for results."
echo "============================================================"
