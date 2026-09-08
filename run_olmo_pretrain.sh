#!/bin/bash
# Regenerate the OLMo-2-1B pretraining checkpoints used by the tables in
# analysis/training_tables.py: 1000 examples each, written 5 records per file,
# so 200 files per checkpoint.
#
# --checkpoint is the HF revision; without it every row would load the default main
# revision, i.e. five identical models.
source env.sh

DATA_NAME="mandarjoshi/trivia_qa"
NUM_EXAMPLES=${NUM_EXAMPLES:-1000}
TEMPERATURE=0.0
GPUS=(4 5 6)

ROWS=(
"allenai/OLMo-2-0425-1B,train,stage1-step480000-tokens1007B"
"allenai/OLMo-2-0425-1B,train,stage1-step1020000-tokens2140B"
"allenai/OLMo-2-0425-1B,train,stage1-step1840000-tokens3859B"
"allenai/OLMo-2-0425-1B,train,stage2-ingredient1-step23852-tokens51B"
"allenai/OLMo-2-0425-1B,train,stage2-ingredient2-step23852-tokens51B"
)

i=0
for row in "${ROWS[@]}"; do
  IFS=',' read -r MODEL_NAME SPLIT CHECKPOINT <<<"$row"
  TS=$(date +%Y%m%d_%H%M%S)
  GPU=${GPUS[$((i % ${#GPUS[@]}))]}
  OUT_DIR="results_vqa/${TS}_temp${TEMPERATURE}_$(basename $MODEL_NAME)_${SPLIT}_${CHECKPOINT}"
  mkdir -p "$OUT_DIR"
  echo "=== [$(date +%H:%M:%S)] gpu$GPU  $CHECKPOINT -> $OUT_DIR"

  CUDA_VISIBLE_DEVICES=$GPU python run/run_generation.py \
    --data_name "$DATA_NAME" --sample false --split "$SPLIT" \
    --temperature "$TEMPERATURE" --calibrate false \
    --model_name "$MODEL_NAME" --checkpoint "$CHECKPOINT" \
    --num_examples "$NUM_EXAMPLES" \
    --out_dir "$OUT_DIR" --timestemp "$TS" \
    > "$OUT_DIR/output.txt" 2>&1 &

  i=$((i+1))
  sleep 2
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
echo "=== all runs finished ==="
for d in results_vqa/*OLMo-2-0425-1B_train_stage*; do
  echo "$(ls $d/results_*.json 2>/dev/null | grep -v vllm | wc -l) files  $d"
done
