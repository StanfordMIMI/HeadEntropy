#!/bin/bash
# ./run_tqa.sh > run_tqa.log 2>&1
source env.sh

DATA_NAME="mandarjoshi/trivia_qa"
RESULTS_DIR="results_vqa"

SAMPLE="false"
TEMPERATURE=0.0
CALIBRATE="false"
THINKING=true
START_HF=0
RESUME_HF=true   # skip records already written to OUT_DIR and carry on from there
export CUDA_VISIBLE_DEVICES=5

# smallest first, so a mistake surfaces before the 32B spends hours on it
ROWS=(
"Qwen/Qwen3-1.7B,validation"
"meta-llama/Llama-3.2-3B-Instruct,validation"
"Qwen/Qwen3-8B,validation"
"meta-llama/Llama-3.1-8B-Instruct,validation"
"google/gemma-4-12b-it,validation"
"Qwen/Qwen3-32B,validation"
"Qwen/Qwen3-1.7B,train"
"meta-llama/Llama-3.2-3B-Instruct,train"
"Qwen/Qwen3-8B,train"
"meta-llama/Llama-3.1-8B-Instruct,train"
"google/gemma-4-12b-it,train"
"Qwen/Qwen3-32B,train"
)

for row in "${ROWS[@]}"; do
  IFS=',' read -r MODEL_NAME SPLIT <<<"$row"

  OUT_DIR=$(run_dir "$RESULTS_DIR" "$TEMPERATURE" "$(basename "$MODEL_NAME")" "$SPLIT")
  TS=$(date +%Y%m%d_%H%M%S)   # id of this invocation, names the files it writes in OUT_DIR
  mkdir -p "$OUT_DIR"

  echo "=== [$(date +%H:%M:%S)] $MODEL_NAME ($SPLIT) -> $OUT_DIR ==="

  if python run/run_generation.py \
    --data_name "$DATA_NAME" \
    --sample "$SAMPLE" \
    --split "$SPLIT" \
    --temperature "$TEMPERATURE" \
    --calibrate "$CALIBRATE" \
    --model_name "$MODEL_NAME" \
    --out_dir "$OUT_DIR" \
    --timestemp "$TS" \
    --start_hf "$START_HF" \
    --thinking "$THINKING" \
    --resume_hf "$RESUME_HF"; then
    echo "OK: $MODEL_NAME ($SPLIT)"
  else
    echo "FAILED: $MODEL_NAME ($SPLIT)"
  fi
done

echo "=== all runs finished ==="
