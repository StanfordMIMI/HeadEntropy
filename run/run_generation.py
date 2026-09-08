import argparse
import json
import os
import sys
import types
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from run.utils.math_evaluation import math_exact_match_score
from run.utils.triviaqa_evaluation import (
    exact_match_score,
    metric_max_over_ground_truths,
)
from run.utils.utils_build_prompts import (
    build_fever_prompt,
    build_hotpot_prompt,
    build_ind_prompt,
    build_math_prompt,
    build_triviaqa_prompt,
    extract_boxed_math,
)
from run.utils.utils_generation import get_messages, scan_processed, send_to_hf, send_to_vllm

def checking_raw_performance(generated_data, data_name=None):
    # MATH answers are latex, where "0.5" and "\\frac{1}{2}" are the same answer, so it
    # is scored on mathematical equality instead of on the string
    if data_name == "EleutherAI/hendrycks_math":
        metric, lower = math_exact_match_score, False
    else:
        metric, lower = exact_match_score, True
    exact_match_list = []
    for item in tqdm(generated_data):
        ground_truth_answers = item["ground_truth_answers"]
        answer = item["model_answer"]
        if lower:
            ground_truth_answers = [ans.lower() for ans in ground_truth_answers]
            answer = answer.lower()
        exact_match = metric_max_over_ground_truths(metric, answer, ground_truth_answers)
        exact_match_list.append(exact_match)
    accuracy = sum(exact_match_list) / len(exact_match_list)
    print(f"Raw Exact Match: {accuracy:.4f}")
    return accuracy

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--sample', type=str2bool, default=False)
    parser.add_argument('--split', type=str)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--calibrate', type=str2bool, default=False)
    parser.add_argument('--start', type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--tok_name', type=str, default=None)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--timestemp', type=str)
    parser.add_argument('--start_hf', type=int, default=0)
    parser.add_argument('--thinking', type=str2bool, default=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    # number of completions sampled per question (vllm SamplingParams.n)
    parser.add_argument('--num_samples', type=int, default=1)
    # nucleus sampling cutoff (vllm SamplingParams.top_p); 1.0 disables it
    parser.add_argument('--top_p', type=float, default=1.0)
    # skip records already written to out_dir and continue where the last run stopped
    parser.add_argument('--resume_hf', type=str2bool, default=False)
    # how many dataset examples to keep after shuffling
    parser.add_argument('--num_examples', type=int, default=30000)
    # generation budget per question. Kept at 768 so runs stay comparable with the
    # existing results; raise it for models whose thinking does not fit in that.
    parser.add_argument('--max_tokens', type=int, default=768)

    args = parser.parse_args()

    print("Loading", args.model_name)
    print("Timestamp:", args.timestemp)
    print("Sample:", args.sample)
    print("Temperature:", args.temperature)
    print("Thinking is set to: ", args.thinking)
    print("Samples per question:", args.num_samples)
    print("Top-p:", args.top_p)
    print("Number of examples:", args.num_examples)
    print("Max generated tokens:", args.max_tokens)
    # vllm batch size
    batch_size = 10000

    # MATH answers are latex: they are extracted and compared as mathematics, not as
    # the 1-3 words the other datasets ask for
    is_math = args.data_name == "EleutherAI/hendrycks_math"

    out_dir_name = args.out_dir
    if not bool(args.thinking):
        out_dir_name += f"_no_thinking"

    print("Output directory:", out_dir_name)
    
    if args.tok_name is None:
        args.tok_name = args.model_name
    print("args.tok_name: ", args.tok_name)

    processor = AutoTokenizer.from_pretrained(args.tok_name)
    data_part = None
    split = args.split
    if args.data_name == "mandarjoshi/trivia_qa":   
        data_part = "rc.nocontext"
    elif args.data_name == "hotpotqa/hotpot_qa":
        data_part = "distractor"
    elif args.data_name == 'openlifescienceai/medmcqa':
        if args.split == "validation":
            split = "validation"
    elif args.data_name == "EleutherAI/hendrycks_math":
        data_part = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
    elif args.data_name == "pietrolesci/nli_fever":
        if args.split == "validation":
            split = "dev"
    else:
        raise ValueError(f"Unknown data_name: {args.data_name}")


    if isinstance(data_part, list):
        from datasets import concatenate_datasets
        splits = [
            load_dataset(
                args.data_name,
                split=split,
                name=part,
            )
            for part in data_part
        ]
        data = concatenate_datasets(splits)
    else:
        data = load_dataset(
            args.data_name,
            split=split,
            name=data_part,
        )
    print("Dataset structure:")
    print(f"Dataset size: {len(data)}")
    print(f"First item keys: {data[0].keys()}")
    print(f"First item: {data[0]}")
    print(f"Second item: {data[1]}")
    print(f"Third item: {data[2]}")

    data = data.shuffle(seed=42)
    data = data.select(range(min(args.num_examples, len(data))))
    if args.data_name == "mandarjoshi/trivia_qa":   
        build_prompt = build_triviaqa_prompt
        ground_truth_answers = [item.get("answer", {}).get("normalized_aliases", {}) for item in data]
        data_part = "rc.nocontext"
    elif args.data_name == "hotpotqa/hotpot_qa":
        build_prompt = build_hotpot_prompt
        ground_truth_answers =  [[item.get("answer")] for item in data]
        data_part = "distractor"
    elif args.data_name == 'openlifescienceai/medmcqa':
        build_prompt = build_ind_prompt
        options = ["opa", "opb", "opc", "opd"]
        answers = [options[int(item.get("cop"))] for item in data]
        ground_truth_answers =  [[item.get(answer)] for item, answer in zip(data, answers)]
    elif args.data_name == "EleutherAI/hendrycks_math":
        build_prompt = build_math_prompt
        # a solution that boxes nothing has no reference answer to score against, so the
        # item is dropped rather than counted as a miss for every model
        boxed = [extract_boxed_math(item) for item in data]
        keep = [i for i, answer in enumerate(boxed) if answer is not None]
        if len(keep) < len(data):
            print(f"Dropping {len(data) - len(keep)} items with no boxed answer")
            data = data.select(keep)
        ground_truth_answers = [[boxed[i]] for i in keep]
    elif args.data_name == "pietrolesci/nli_fever":
        build_prompt = build_fever_prompt
        ground_truth_answers = [[item.get("fever_gold_label")] for item in data]
    else:
        raise ValueError(f"Unknown data_name: {args.data_name}")
    
    print("Number of answers: ",len(ground_truth_answers))
    messages = get_messages(data, processor, build_prompt, enable_thinking=args.thinking)
    print("Number of messages: ",len(messages))
    data = data.add_column("messages", messages)
    data = data.add_column("ground_truth_answers", ground_truth_answers)

    messages_chunked = [
        data.select(range(start, min(start + batch_size, len(data))))
        for start in range(0, len(data), batch_size)
    ]
        
    os.makedirs(out_dir_name, exist_ok=True)

    # 1. generate the answers with vllm for efficiency reasons
    vllm_outputs = [os.path.join(out_dir_name, f"results_vllm_{i}.json") for i in range(len(messages_chunked))]
    print("vllm_outputs", vllm_outputs)
    if args.checkpoint == "None":
        args.checkpoint = None
    if all([os.path.exists(vllm_output) for vllm_output in vllm_outputs]):
        print("Skipping vllm generation because all files already exist")
    else:
        print("Generating with vllm")
        max_input_len = max(len(m) for m in data["messages"])
        max_model_len = max_input_len + args.max_tokens  # input + output budget
        from vllm import LLM
        model_vllm = LLM(
            model=args.model_name,
            revision=args.checkpoint,
            tokenizer=args.tok_name,
            dtype=torch.bfloat16,
            tensor_parallel_size=1,
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            max_model_len=max_model_len,
        )
        vllm_outputs = send_to_vllm(
            model_vllm,
            data,
            out_dir_name,
            batch_size=batch_size,
            temperature=args.temperature,
            num_samples=args.num_samples,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            is_math=is_math,
        )
        print("Done with generation with vllm")
        del model_vllm
    torch.cuda.empty_cache()
    generated_data = []
    for path in vllm_outputs:
        with open(path, "r") as f:
            batch = json.load(f)
        generated_data.extend(batch)
    print("Number of generated data: ", len(generated_data))
    check_performance = checking_raw_performance(generated_data, args.data_name)
    print("Raw performance checked: ", check_performance)
    using_generated_data = generated_data[args.start_hf:]
    print("Number of data to be processed with hf: ", len(using_generated_data))

    # 2. pass the question and answer through the model again to get the attention maps 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, 
        device_map= "auto", # {"": 0} if torch.cuda.is_available() else None, # "auto", # 
        attn_implementation="eager", 
        # cache_dir=f"{root}/hf_model_tmp_new",
        revision=args.checkpoint,
        trust_remote_code=True,
    )
    skip_idxs, count_start = set(), 0
    if args.resume_hf:
        skip_idxs, count_start = scan_processed(out_dir_name, args.timestemp)
        remaining = len(using_generated_data) - len([i for i in skip_idxs if i >= args.start_hf])
        print(f"Resuming: {len(skip_idxs)} records already on disk, {remaining} left to process")

    send_to_hf(
            model=model,
            processor=processor,
            generated_data=using_generated_data,
            timestemp=args.timestemp,
            out_dir=out_dir_name,
            start_idx=args.start_hf,
            skip_idxs=skip_idxs,
            count_start=count_start,
            is_math=is_math,
        )
    del model
    torch.cuda.empty_cache()

    print("Done with processing with hf")

if __name__ == "__main__":
    main()
