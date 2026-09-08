#!/bin/bash

python run/run_analyses.py \
  --out_prefix_train "$PATH_TRAIN" \
  --out_prefix_val "$PATH_VAL" \
  --model_name "$MODEL_NAME" \
  --med_generalization "$MED_GENERALIZATION" \
  --hot_generalization "$HOT_GENERALIZATION" \
  --math_generalization "$MATH_GENERALIZATION" \
  --fever_generalization "$FEVER_GENERALIZATION" \
  --compute_ci True \
  --exclude_truncated_train "${EXCLUDE_TRUNCATED_TRAIN:-True}" \
  --exclude_truncated_eval "${EXCLUDE_TRUNCATED_EVAL:-False}" \
  --semantic_section trace_jacobian_answer_end \
  # --debug True \
  # --thinking False \
  # --no_scaling True 
    # --temp_generalization "$TEMP_GENERALIZATION" \
#   # --temp_generalization "$TEMP_GENERALIZATION" \
