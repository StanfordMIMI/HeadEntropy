#!/bin/bash
# Run one analysis: ./eval.sh <dataset> <model> [variant]
#
#   ./eval.sh tqa 8B_q                # TriviaQA, Qwen3-8B; variant defaults to "answer"
#   ./eval.sh hqa 8B_l answer_debug   # HotpotQA, Llama-3.1-8B-Instruct, 200-example smoke test
#   ./eval.sh --list                  # what the keys mean and which runs exist
#
# <dataset> picks the training set (and its validation split); MedMCQA, HotpotQA, MATH
# and FEVER are used as transfer sets whenever a run for <model> exists there. <model> is
# a key from --list (a full model name such as Qwen3-8B is accepted too). [variant]
# selects eval_variants/<variant>.sh, which holds the python invocation.
source env.sh

VARIANT_DIR="./eval_variants"
TEMPERATURE="0.0"

# dataset key -> results directory, directory suffix, long name
dataset_info() {
    case "$1" in
        tqa)   echo "results_vqa    ''        TriviaQA" ;;
        hqa)   echo "results_hot    _hot      HotpotQA" ;;
        indqa) echo "results_indmed _indmed   MedMCQA" ;;
        *)     return 1 ;;
    esac
}

# model key -> model short name (basename of the HuggingFace id)
model_name() {
    case "$1" in
        1.7B) echo "Qwen3-1.7B" ;;
        3B)   echo "Llama-3.2-3B-Instruct" ;;
        8B_q) echo "Qwen3-8B" ;;
        8B_l) echo "Llama-3.1-8B-Instruct" ;;
        12B)  echo "gemma-4-12b-it" ;;
        32B)  echo "Qwen3-32B" ;;
        *)    echo "$1" ;;     # not a key: taken as a model name as is
    esac
}
MODEL_KEYS="1.7B 3B 8B_q 8B_l 12B 32B"

# "tqa hqa indqa" restricted to the datasets that have a train and validation run
train_sets_for() {
    local d results_dir suffix long_name out=""
    for d in tqa hqa indqa; do
        read -r results_dir suffix long_name <<<"$(dataset_info "$d")"
        [ "$suffix" = "''" ] && suffix=""
        [ -d "$(run_dir "$results_dir" "$TEMPERATURE" "$1" train "$suffix")" ] &&
        [ -d "$(run_dir "$results_dir" "$TEMPERATURE" "$1" validation "$suffix")" ] && out="$out $d"
    done
    echo "${out# }"
}

# transfer sets that have a run for the model
transfer_sets_for() {
    local out=""
    [ -d "$(run_dir results_indmed "$TEMPERATURE" "$1" validation _indmed)" ] && out="$out MedMCQA"
    [ -d "$(run_dir results_hot    "$TEMPERATURE" "$1" validation _hot)" ]    && out="$out HotpotQA"
    [ -d "$(run_dir results_math2  "$TEMPERATURE" "$1" test)" ]               && out="$out MATH"
    [ -d "$(run_dir results_fever  "$TEMPERATURE" "$1" validation)" ]         && out="$out FEVER"
    echo "${out# }"
}

list_options() {
    echo "usage: $0 <dataset> <model> [variant]"
    echo
    echo "datasets (<dataset>): the probe is trained on its train split and evaluated on its validation split"
    local d results_dir suffix long_name
    for d in tqa hqa indqa; do
        read -r results_dir suffix long_name <<<"$(dataset_info "$d")"
        printf "    %-6s %s\n" "$d" "$long_name"
    done
    echo "    MATH and FEVER have no train split; they and the other datasets are transfer sets,"
    echo "    used automatically when a run for the model exists there"
    echo
    printf "%-40s %-16s %s\n" "models (<model>):" "train sets found" "transfer sets found"
    local k name
    for k in $MODEL_KEYS; do
        name=$(model_name "$k")
        printf "    %-6s %-24s     %-16s %s\n" "$k" "$name" "$(train_sets_for "$name")" "$(transfer_sets_for "$name")"
    done
    echo
    echo "variants (<variant>, default answer):"
    local f
    for f in "$VARIANT_DIR"/*.sh; do
        echo "    $(basename "$f" .sh)"
    done
}

if [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
    list_options
    exit 0
fi

if [ $# -lt 2 ]; then
    echo "usage: $0 <dataset> <model> [variant]    (see $0 --list)" >&2
    exit 2
fi

DATASET=$1
MODEL_NAME=$(model_name "$2")
VARIANT=${3:-answer}

if ! info=$(dataset_info "$DATASET"); then
    echo "unknown dataset: $DATASET (one of tqa, hqa, indqa)" >&2
    exit 1
fi
read -r RESULTS_DIR SUFFIX _ <<<"$info"
[ "$SUFFIX" = "''" ] && SUFFIX=""

VARIANT_CONFIG="$VARIANT_DIR/${VARIANT}.sh"
if [ ! -f "$VARIANT_CONFIG" ]; then
    echo "no such variant: $VARIANT" >&2
    list_options >&2
    exit 1
fi

# a transfer set that has no run for this model is passed as "", which
# run_analyses.py treats as "skip"
existing() { [ -d "$1" ] && echo "$1" || echo ""; }

PATH_TRAIN=$(run_dir "$RESULTS_DIR" "$TEMPERATURE" "$MODEL_NAME" train "$SUFFIX")
PATH_VAL=$(run_dir "$RESULTS_DIR" "$TEMPERATURE" "$MODEL_NAME" validation "$SUFFIX")
for p in "$PATH_TRAIN" "$PATH_VAL"; do
    if [ ! -d "$p" ]; then
        echo "no generation run found: $p" >&2
        echo "run the matching run_*.sh first, or see $0 --list" >&2
        exit 1
    fi
done

MED_GENERALIZATION=$(existing "$(run_dir results_indmed "$TEMPERATURE" "$MODEL_NAME" validation _indmed)")
HOT_GENERALIZATION=$(existing "$(run_dir results_hot "$TEMPERATURE" "$MODEL_NAME" validation _hot)")
MATH_GENERALIZATION=$(existing "$(run_dir results_math2 "$TEMPERATURE" "$MODEL_NAME" test)")
FEVER_GENERALIZATION=$(existing "$(run_dir results_fever "$TEMPERATURE" "$MODEL_NAME" validation)")
TEMP_GENERALIZATION=$(existing "$(run_dir results_vqa 1.0 "$MODEL_NAME" validation)")

echo "==> ${DATASET} ${2} (${MODEL_NAME}) ${VARIANT}"
echo "    train:  $PATH_TRAIN"
echo "    val:    $PATH_VAL"
echo "    med:    ${MED_GENERALIZATION:-skipped}"
echo "    hotpot: ${HOT_GENERALIZATION:-skipped}"
echo "    math:   ${MATH_GENERALIZATION:-skipped}"
echo "    fever:  ${FEVER_GENERALIZATION:-skipped}"
source "$VARIANT_CONFIG"
