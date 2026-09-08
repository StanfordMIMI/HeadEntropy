#!/bin/bash
# Same analysis as answer.sh on a 200-example subset, for a quick end-to-end check.

python run/run_analyses.py \
  --out_prefix_train "$PATH_TRAIN" \
  --out_prefix_val "$PATH_VAL" \
  --model_name "$MODEL_NAME" \
  --med_generalization "$MED_GENERALIZATION" \
  --hot_generalization "$HOT_GENERALIZATION" \
  --math_generalization "$MATH_GENERALIZATION" \
  --fever_generalization "$FEVER_GENERALIZATION" \
  --exclude_truncated_train "${EXCLUDE_TRUNCATED_TRAIN:-True}" \
  --exclude_truncated_eval "${EXCLUDE_TRUNCATED_EVAL:-False}" \
  --semantic_section trace_jacobian_answer_end \
  --debug True
