#!/bin/bash
# ./run_mathqa.sh > run_mathqa.log 2>&1
source env.sh

DATA_NAME="EleutherAI/hendrycks_math"
RESULTS_DIR="results_math2"

SAMPLE="false"
TEMPERATURE=0.0
CALIBRATE="false"
START_HF=0
RESUME_HF=true
export CUDA_VISIBLE_DEVICES=5

# Twice the usual budget: the thinking models spend all 768 tokens inside <think> on
# MATH and 75-86% of their generations never reach an "Answer:" line. The Llama rows
# answer in ~50 tokens and are unaffected.
MAX_TOKENS=1024

# smallest first, so a mistake surfaces before the 32B spends hours on it
ROWS=(
"Qwen/Qwen3-1.7B,test"
"Qwen/Qwen3-8B,test"
"google/gemma-4-12b-it,test"
"Qwen/Qwen3-32B,test"
"meta-llama/Llama-3.2-3B-Instruct,test"
"meta-llama/Llama-3.1-8B-Instruct,test"
)

for row in "${ROWS[@]}"; do
  IFS=',' read -r MODEL_NAME SPLIT <<<"$row"

  OUT_DIR=$(run_dir "$RESULTS_DIR" "$TEMPERATURE" "$(basename "$MODEL_NAME")" "$SPLIT")
  TS=$(date +%Y%m%d_%H%M%S)   # id of this invocation, names the files it writes in OUT_DIR
  mkdir -p "$OUT_DIR"

  echo "=== [$(date +%H:%M:%S)] $MODEL_NAME ($SPLIT, ${MAX_TOKENS} tok) -> $OUT_DIR ==="

  if python run/run_generation.py \
    --data_name "$DATA_NAME" \
    --sample "$SAMPLE" \
    --split "$SPLIT" \
    --temperature "$TEMPERATURE" \
    --calibrate "$CALIBRATE" \
    --model_name "$MODEL_NAME" \
    --out_dir "$OUT_DIR" \
    --timestemp "$TS" \
    --max_tokens "$MAX_TOKENS" \
    --start_hf "$START_HF" \
    --resume_hf "$RESUME_HF"; then
    echo "OK: $MODEL_NAME ($SPLIT)"
  else
    echo "FAILED: $MODEL_NAME ($SPLIT)"
  fi
done

echo "=== all runs finished ==="
