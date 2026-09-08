#!/bin/bash
# Shared environment for run_*.sh and eval.sh. Sourced, not executed.

# keep every cache inside the repository
export HF_DATASETS_CACHE="./hf_datasets_tmp"
export HUGGINGFACE_HUB_CACHE="./hf_model_tmp"
export HF_HOME="./hf_home"
export TORCH_COMPILE_CACHE_DIR="./tmp_cache/torch_compile_cache"
export XDG_CACHE_HOME="./tmp_cache"
export TMPDIR="/dev/shm/$USER"
export PYTHONPATH=".:$PYTHONPATH"
mkdir -p "$TMPDIR"

# run_dir <results_dir> <temperature> <model short name> <split> [suffix]
#
# Directory of one generation run, e.g. run_dir results_vqa 0.0 Qwen3-8B validation.
# Matches both the plain name written by the run scripts (temp0.0_Qwen3-8B_validation)
# and the timestamped ones of older runs (20250627_..._temp0.0_Qwen3-8B_validation).
# If several exist the one with the most feature files wins, so a resumed run and the
# analysis both land on the directory that holds the data, not on an empty duplicate.
# Prints the plain name if none exists.
run_dir() {
    local best="" best_n=-1 d n
    for d in "$1"/*"temp$2_$3_$4$5"; do
        [ -d "$d" ] || continue
        n=$(ls "$d"/results_*.json 2>/dev/null | grep -vc vllm)
        if [ "$n" -ge "$best_n" ]; then best=$d; best_n=$n; fi
    done
    echo "${best:-$1/temp$2_$3_$4$5}"
}
