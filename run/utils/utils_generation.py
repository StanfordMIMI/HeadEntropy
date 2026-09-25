import sys, types
import numpy as np
import torch
import ipdb
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    dummy = types.ModuleType("cv2")
    sys.modules["cv2"] = dummy
import glob
import math
import os
import json
import re
from tqdm import tqdm
import torch.nn.functional as F
from transformers import DataCollatorWithPadding
from run.utils.triviaqa_evaluation import exact_match_score, metric_max_over_ground_truths, f1_score
from run.utils.utils_icr_score import ICRScore
from run.utils.utils_build_prompts import _last_boxed
from run.utils.math_evaluation import math_exact_match_score
# from run.utils.utils_analyse import make_histograms #, make_scatter_with_lines, make_mean_variance_scatter
from typing import Any, Mapping, Optional, Dict

def score_to_temperature(score, t_min=0.1, t_max=1.9):
    temp = t_min + (1.0 - score) * (t_max - t_min)
    temp = np.round(temp, 1)

    if isinstance(temp, np.ndarray):
        return float(temp.item())
    return float(temp)

def full_attn(gen_out, prompt_len: int):
    step0 = gen_out.attentions[0]  # tuple of layers
    layers = len(step0)
    heads = step0[0].shape[1]
    gen_steps = len(gen_out.attentions) - 1  # tokens actually generated
    T0 = prompt_len  # + 1  # length after first step
    T = prompt_len + gen_steps + 1  # final length
    full = torch.zeros(layers, heads, T, T, device=step0[0].device)

    for l, a in enumerate(step0):  # a: [heads, T0, T0]
        full[l, :, :T0, :T0] = a

    for s in range(1, gen_steps + 1):
        row_idx = prompt_len + s  # which row to write
        step_att = gen_out.attentions[s]  # tuple of layers
        seq_len = prompt_len + s  # + 1

        for l, a in enumerate(step_att):  # a: [heads, 1, seq_len]
            full[l, :, row_idx, :seq_len] = a.squeeze(0).squeeze(1)

    return full  # [layers, heads, T, T]

def row_trace_jacobian(full):
    try:
        prob = full.float()
        return 1 - (prob ** 2).sum(dim=-1) 
    except:
        entropy = torch.zeros(full.shape[0], full.shape[1], full.shape[2], device=full.device)
        for l in range(full.shape[0]):
            for h in range(full.shape[1]):
                prob = full[l,h].float()  # [heads, T, T]
                H_row = 1 - (prob ** 2).sum(dim=-1) 
                entropy[l, h] = H_row
        return entropy

def make_collate_fn(processor):
    base_collator = DataCollatorWithPadding(tokenizer=processor, padding="longest")
    def collate_fn(batch):
        batch = [x for x in batch if isinstance(x, dict)]
        if len(batch) == 0:
            return None
        return base_collator(batch)
    return collate_fn

def get_messages(data, processor, prompt_func, enable_thinking=True):
    messages = []
    has_chat_template = processor.chat_template is not None

    for item in tqdm(data):
        system_message, prompt = prompt_func(item)
        if has_chat_template:
            message = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ]
            text = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        else:
            text = system_message + "\n\n" + prompt
        messages.append(text)
    return messages   

def send_to_vllm(
    model_vllm,
    data,
    out_dir,
    batch_size: int = 1000,
    temperature: int = 0,
    num_samples: int = 1,
    top_p: float = 1.0,
    max_tokens: int = 768,
    is_math: bool = False,
):
    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # chunk the messages into batches of size `batch_size`, ensuring indices stay within bounds
    messages_chunked = [
        data.select(range(start, min(start + batch_size, len(data))))
        for start in range(0, len(data), batch_size)
    ]

    paths_to_vllm_out = []
    for i, messages_chunk in tqdm(enumerate(messages_chunked)):
        path_to_vllm_out = f"{out_dir}/results_vllm_{i}.json"
        if os.path.exists(path_to_vllm_out):
            print(f"Skipping {path_to_vllm_out} because it already exists")
            paths_to_vllm_out.append(path_to_vllm_out)
            continue

        vllm_out = model_vllm.generate(messages_chunk["messages"], sampling_params)
        # save the vllm_out to a file

        print("Making outputs_all_list from vllm_out")
        # flat list: with num_samples > 1 each question contributes one record per
        # sample, all sharing question_id and distinguished by sample_idx
        outputs_all_list = [{
            # use existing question_id if present, otherwise fall back to a running index within the dataset
            "question_id": batch["id"] if "id" in batch.keys() else batch.get("question_id", (i * batch_size) + idx),
            # only a multi-sample run needs it: at n=1 it is always 0, and emitting it
            # there would leave single-sample results inconsistent with every run
            # recorded before the key existed
            **({"sample_idx": sample_idx} if num_samples > 1 else {}),
            "prompt": batch["messages"],
            "output": sample.text,
            "question": batch["messages"].split("Question: ")[-1].split("\n")[0],
            "model_answer": (extract_math_answer if is_math else extract_answer)(sample.text),
            "model_confidence": extract_confidence(sample.text),
            "prompt_token_ids": ex.prompt_token_ids,
            "answer_token_ids": sample.token_ids,
            "ground_truth_answers": batch["ground_truth_answers"],
        } for idx, (ex, batch) in tqdm(enumerate(zip(vllm_out, messages_chunk)))
          for sample_idx, sample in enumerate(ex.outputs)]

        with open(path_to_vllm_out, "w") as f:
            json.dump(outputs_all_list, f, indent=2)
        paths_to_vllm_out.append(path_to_vllm_out)

    return paths_to_vllm_out

def _token_strings(input, processor, strip_chars=("Ä", "Ġ", "Ċ")):
    """Token strings for ``input``, with any id the tokenizer cannot name as "".

    A model can generate an embedding row the tokenizer has no entry for -- Qwen3's
    matrix is 151936 wide where its tokenizer names 151669 -- and
    convert_ids_to_tokens hands back None for it. Such a token is never one of the
    markers looked for below, so it is read as empty rather than taking the whole
    record down.
    """
    cleaned = []
    for token in processor.convert_ids_to_tokens(input):
        if token is None:
            cleaned.append("")
            continue
        for char in strip_chars:
            token = token.replace(char, "")
        cleaned.append(token)
    return cleaned


def compute_token_idx_olmo(input, processor):
    assert "olmo" in processor.name_or_path.lower(), "Processor is not Olmo but token idx function is set to olmo."
    tokens = _token_strings(input, processor)
    assert "Please" in tokens
    assert "Question" in tokens

    system_end = next(
        (
            i
            for i, t in enumerate(tokens)
            if t and (t.strip('"').startswith("Please"))
        ),
        len(tokens),
    )

    template_end = next(
        (
            i
            for i, t in enumerate(tokens[system_end:], start=system_end)
            if (tokens[i].strip('"') == ']' or tokens[i].strip('"') == '>') and tokens[i + 1] == 'Question'
        ),
        len(tokens),
    )

    question_end = next(
    (
        i
        for i in range(template_end, len(tokens) - 1)
        if tokens[i].strip('"') == "?" and tokens[i + 1] == ''    ),
    len(tokens),
    )
    
    question_end = len(tokens) - 10 if question_end == len(tokens) else question_end 

    answer_end = len(tokens)
    assert system_end <= template_end <= question_end <= answer_end <= len(tokens)

    return {
        "system_end": system_end,
        "template_end": template_end,
        "question_end": question_end,
        "answer_end": answer_end,
    }


def compute_token_idx_llama(input, processor):
    assert "llama" in processor.name_or_path.lower(), "Processor is not Llama but token idx function is set to qwen."
    tokens = _token_strings(input, processor)
    system_end = next(
        (
            i
            for i, t in enumerate(tokens)
            if t and (t.strip('"').upper().startswith("PLEA") or t.strip('"').upper().startswith(".PLEA"))
        ),
        len(tokens),
    )

    template_end = next(
        (
            i
            for i, t in enumerate(tokens[system_end:], start=system_end)
            if tokens[i].strip('"') == "user" and tokens[i + 1] == "<|end_header_id|>"
        ),
        len(tokens),
    )

    question_end = next(
    (
        i
        for i in range(template_end, len(tokens) - 1)
        if tokens[i].strip('"') == "assistant" and tokens[i + 1] == "<|end_header_id|>"
    ),
    len(tokens),
    )
    answer_end = next(
        ( 
            i
            for i, t in enumerate(tokens[question_end:], start=question_end)
            if t and (t.strip('"').startswith('Cert'))
        ),
        len(tokens),
    )

    assert system_end <= template_end <= question_end <= answer_end <= len(tokens)
    certainty_end = len(tokens)

    return {
        "system_end": system_end,
        "question_end": question_end,
        "template_end": template_end,
        "answer_end": answer_end,
        "certainty_end": certainty_end
    }

def compute_token_idx_qwen(input, processor):
    assert "qwen" in processor.name_or_path.lower(), "Processor is not Qwen but token idx function is set to qwen."
    tokens = _token_strings(input, processor)
    system_end = next(
        (
            i
            for i, t in enumerate(tokens)
            if t and (t.strip('"').upper().startswith("PLEA") or t.strip('"').upper().startswith(".PLEA"))
        ),
        len(tokens),
    )

    template_end = next(
        (
            i
            for i, t in enumerate(tokens[system_end:], start=system_end)
            if tokens[i].strip('"') == "<|im_start|>" and tokens[i + 1] == "user"
        ),
        len(tokens),
    )
    question_end = next(
    (
        i
        for i in range(template_end, len(tokens) - 1)
        if tokens[i].strip('"') == "<|im_start|>" and tokens[i + 1] == "assistant"
    ),
    len(tokens),
    )

    think_chunks = {}

    end_of_think_token = "</think>"
    if not end_of_think_token in tokens:
        end_of_think_token = "Answer"
    think_end = next(
        (
            i
            for i, t in enumerate(tokens[question_end:], start=question_end)
            if t and (t.strip('"').startswith(end_of_think_token))
        ),
        len(tokens),
    )  
    count = 1
    if think_end - question_end > 100:
        # chunk up in 100 thinking token chunks
        start = question_end + 100
        for i in range(start, think_end, 100):
            think_chunks[f"think_chunk_{count}"] = i
            count += 1
    think_chunks[f"think_chunk_{count}"] = think_end

    if not think_end == len(tokens):
        answer_end = next(
            (
                i
                for i, t in enumerate(tokens[think_end:], start=think_end)
                if t and (t.strip('"').startswith('Cert'))# or t.strip('"').startswith(':'))
            ),
            len(tokens),
        )
    else:
        answer_end = len(tokens)

    assert system_end <= template_end <= question_end <= answer_end <= len(tokens)
    certainty_end = len(tokens)

    return {
        "system_end": system_end,
        "question_end": question_end,
        "template_end": template_end,
        "answer_end": answer_end,
        "certainty_end": certainty_end,
        **think_chunks,
    }


def compute_token_idx_gemma4(input, processor):
    name = processor.name_or_path.lower()
    assert "gemma-4" in name, \
        f"Expected Gemma 4, got {processor.name_or_path}"

    tokens = _token_strings(input, processor, strip_chars=("▁",))

    def clean(t):
        return t.strip('"') if t else t

    # System content boundary — PLEA heuristic
    system_end = next(
        (i for i, t in enumerate(tokens)
         if t and (clean(t).upper().startswith("PLEA")
                   or clean(t).upper().startswith(".PLEA"))),
        len(tokens),
    )

    # User turn: <|turn> followed by "user" within next 2 tokens
    template_end = next(
        (i for i, t in enumerate(tokens[system_end:], start=system_end)
         if clean(t) == "<|turn>"
         and any(clean(tokens[i + j]).startswith("user") for j in range(1, 3))),
        len(tokens),
    )

    # Model turn: <|turn> followed by "model" within next 2 tokens
    question_end = next(
        (i for i in range(template_end, len(tokens) - 1)
         if clean(tokens[i]) == "<|turn>"
         and any(clean(tokens[i + j]).startswith("model") for j in range(1, 3))),
        len(tokens),
    )

    # ---- Thinking tokens (mirrors Qwen logic) ----
    think_chunks = {}

    end_of_think_token = "<channel|>"
    if end_of_think_token not in tokens:
        end_of_think_token = "Answer"
    think_end = next(
        (i for i, t in enumerate(tokens[question_end:], start=question_end)
         if t and clean(t).startswith(end_of_think_token)),
        len(tokens),
    )

    count = 1
    if think_end - question_end > 100:
        start = question_end + 100
        for i in range(start, think_end, 100):
            think_chunks[f"think_chunk_{count}"] = i
            count += 1
    think_chunks[f"think_chunk_{count}"] = think_end

    # Answer end — "Cert" heuristic
    if not think_end == len(tokens):
        answer_end = next(
            (i for i, t in enumerate(tokens[think_end:], start=think_end)
             if t and clean(t).startswith('Cert')),
            len(tokens),
        )
    else:
        answer_end = len(tokens)
    # ---- End thinking tokens ----

    assert system_end <= template_end <= question_end <= answer_end <= len(tokens)
    certainty_end = len(tokens)

    return {
        "system_end": system_end,
        "template_end": template_end,
        "question_end": question_end,
        "answer_end": answer_end,
        "certainty_end": certainty_end,
        **think_chunks,
    }

def compute_entropy(attentions, token_idx: dict, inputs: dict):
    entropy = row_entropy(attentions)  # [layers, heads, T]
    # Sort token_idx by values
    chunk_ends = list(token_idx.values())
    chunk_keys = list(token_idx.keys())
    return_dict = {}
    # chunk up the entropy
    start = 0
    for i in range(len(chunk_ends)):
        end = chunk_ends[i]
        if end == start:
            return_dict[f"entropy_{chunk_keys[i]}"] = [None for _ in range(entropy.shape[0]*entropy.shape[1])]
            continue
        entropy_slice = entropy[:, :, start:end]
        # normalize by sequence lengths per row, start at 1 not 0
        # denom = torch.log(torch.arange(start+1, end+1, device=entropy_slice.device, dtype=entropy_slice.dtype))
        # entropy_slice_norm = entropy_slice / denom    # [L, H, T]
        # entropy_slice_norm[torch.isnan(entropy_slice_norm)] = 0 # first token always has 0 entropy

        # assert not torch.isnan(entropy_slice_norm).any()
        return_dict[f"entropy_{chunk_keys[i]}"] = entropy_slice.mean(dim=-1).flatten().tolist()
        return_dict[f"entropy_{chunk_keys[i]}_max"] = entropy_slice.max(dim=-1).values.flatten().tolist()
        return_dict[f"entropy_{chunk_keys[i]}_min"] = entropy_slice.min(dim=-1).values.flatten().tolist()
        return_dict[f"entropy_token_{chunk_keys[i]}"] = entropy_slice.flatten(0,1).mean(dim=0).tolist()
        # assert not torch.isnan(entropy_slice_norm.mean(dim=-1).flatten()).any(), print("Start: ", start, " End: ", end)
        start = end
    return return_dict

def compute_trace_jacobian(attentions, token_idx: dict, inputs: dict):
    entropy = row_trace_jacobian(attentions)  # [layers, heads, T]
    # Sort token_idx by values
    chunk_ends = list(token_idx.values())
    chunk_keys = list(token_idx.keys())
    return_dict = {}
    # chunk up the entropy
    start = 0
    for i in range(len(chunk_ends)):
        end = chunk_ends[i]
        if end == start:
            return_dict[f"trace_jacobian_{chunk_keys[i]}"] = [None for _ in range(entropy.shape[0]*entropy.shape[1])]
            continue
        entropy_slice = entropy[:, :, start:end]
        # normalize by sequence lengths per row, start at 1 not 0
        # denom = torch.log(torch.arange(start+1, end+1, device=entropy_slice.device, dtype=entropy_slice.dtype))
        # entropy_slice_norm = entropy_slice / denom    # [L, H, T]
        # entropy_slice_norm[torch.isnan(entropy_slice_norm)] = 0 # first token always has 0 entropy

        # assert not torch.isnan(entropy_slice_norm).any()
        return_dict[f"trace_jacobian_{chunk_keys[i]}"] = entropy_slice.mean(dim=-1).flatten().tolist()
        # return_dict[f"trace_jacobian_{chunk_keys[i]}_max"] = entropy_slice.max(dim=-1).values.flatten().tolist()
        # return_dict[f"trace_jacobian_{chunk_keys[i]}_min"] = entropy_slice.min(dim=-1).values.flatten().tolist()
        # return_dict[f"trace_jacobian_token_{chunk_keys[i]}"] = entropy_slice.flatten(0,1).mean(dim=0).tolist()
        # assert not torch.isnan(entropy_slice_norm.mean(dim=-1).flatten()).any(), print("Start: ", start, " End: ", end)
        start = end
    return_dict[f"trace_jacobian_all"] = entropy.mean(dim=-1).flatten().tolist()
    return return_dict

def compute_lookback_lens(attentions, token_idx):
    # code from https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step01_extract_attns.py#L244
    context_length = token_idx["question_end"]
    num_layers, num_heads, seq_len, _ = attentions.shape
    new_token_length = seq_len - context_length 
    assert new_token_length > 0, f"Context length {context_length} exceeds total sequence length {seq_len}. token_idx {token_idx}"
    # ipdb.set_trace()
    
    lookback_ratio = torch.zeros((num_layers, num_heads, new_token_length))
    
    for i in range(new_token_length):  # iterating over the new tokens
        token_pos = context_length + i 
        for l in range(num_layers):
            attn_on_context = attentions[l, :, token_pos, :context_length].mean(-1)
            attn_on_new_tokens = attentions[l, :, token_pos, context_length:].mean(-1)
            lookback_ratio[l, :, i] = attn_on_context / (attn_on_context + attn_on_new_tokens)
    # https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step03_lookback_lens.py#L311C1-L313C42
    # Reshape: [num_layers, num_heads, num_new_tokens] -> [num_new_tokens, num_layers * num_heads]
    lookback_ratio = lookback_ratio.view(-1, lookback_ratio.shape[2])
    lookback_ratio = lookback_ratio.transpose(0, 1)
    
    # Average across all tokens to get single feature vector
    feature_vector = lookback_ratio.mean(dim=0)
    
    return feature_vector.tolist()  # shape: [num_layers * num_heads]

def compute_logprobs(scores, processor):
    logprobs = []
    for step in scores:
        lp = torch.log_softmax(step, dim=-1)
        probs = lp.exp()

        entropy = -torch.sum(probs * lp.clamp(min=-100)).item()
        # k = 5
        # topk_lp, topk_idx = torch.topk(lp, k)
        gen_idx = step.argmax(dim=-1).item()
        logprobs.append(
            {
                "token": processor.decode([gen_idx]),
                "logprob": lp[gen_idx].item(),
                "entropy": entropy,
                # "top_logprobs": [
                #     {
                #         "token": processor.decode([topk_idx[i].item()]),
                #         "logprob": topk_lp[i].item(),
                #     }
                #     for i in range(k)
                # ],
            }
        )
    return logprobs

# Models often ignore the two-line format and glue the certainty onto the answer
# ("Answer: Pulsar: 90", "Ruddigore\n70"). Left in, the number survives
# normalize_answer as a second token and the exact match always fails.
_CERTAINTY_SUFFIX = re.compile(r"\s*(?:certainty|confidence)\b\s*[:\-]?\s*\d{0,3}\s*%?\s*$", re.I)
_TRAILING_CONFIDENCE = re.compile(r"\s*[:\-]\s*\d{1,3}\s*%?\s*$")
_CONFIDENCE_LABEL = re.compile(r"(?:certainty|confidence)\b\s*[:\-]?\s*(\d{1,3})", re.I)


def extract_answer(text):
    """The answer span of a generation, without any certainty glued onto it.

    Only strips a trailing number when a colon or the word certainty introduces it, so
    answers that are themselves numeric ("1984") or end in one ("Apollo 11") survive.
    """
    if not text:
        return ""
    # Matched from the end, as the split("Answer: ")[-1] this replaced did. A thinking
    # trace restates the prompt's format spec ("Format: Answer: [1-5 words only]") and
    # weighs candidates before it commits, so the first label is the template and the
    # last is the answer. Matching the first cost gemma-4-12b-it HotpotQA 0.71 -> 0.01.
    # [^\S\n] rather than \s so a bare "Answer:" cannot swallow the following line.
    labelled = [m.group(1).strip() for m in re.finditer(r"answer\s*:[^\S\n]*(.*)", text, re.I)]
    if labelled:
        ans = next((a for a in reversed(labelled) if a), "")
    else:
        ans = text.strip()                 # unlabelled generation: take it as written
    if not ans:
        return ""
    ans = ans.splitlines()[0]              # answer is one line; certainty follows on the next
    ans = _CERTAINTY_SUFFIX.sub("", ans)   # "Pulsar Certainty: 90"
    ans = _TRAILING_CONFIDENCE.sub("", ans)  # "Pulsar: 90"
    return ans.strip().strip('"').strip()


# MATH answers are latex, and the two rules above do real damage to it:
# _TRAILING_CONFIDENCE reads "-2" as a bare confidence and deletes the whole answer,
# and cuts "n-1" down to "n" -- 186 of the 191 answers Llama-3.2-3B-Instruct left empty
# on MATH were negative numbers. Only a labelled certainty is stripped here.
_MATH_CERTAINTY_SUFFIX = re.compile(
    r"[\s,;.]*(?:certainty|confidence)\b\s*[:\-]?\s*\d{0,3}\s*%?\s*$", re.I)
_MATH_CERTAINTY_LABEL = re.compile(r"(?:certainty|confidence)\s*[:\-]", re.I)


def _unescaped_dollars(s):
    return len(re.findall(r"(?<!\\)\$", s))


def _latex_balanced(s):
    """Whether every latex group opened in ``s`` is also closed in it."""
    return (s.count("{") <= s.count("}")
            and s.count("\\[") <= s.count("\\]")
            and s.count("\\begin") <= s.count("\\end")
            and _unescaped_dollars(s) % 2 == 0)


def _strip_math_delimiters(s):
    """``$...$``, ``\\[...\\]`` and ``\\(...\\)`` wrapping a whole answer mean nothing."""
    s = s.strip()
    for opener, closer in (("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)")):
        while (s.startswith(opener) and s.endswith(closer)
               and len(s) > len(opener) + len(closer)):
            inner = s[len(opener):-len(closer)]
            # "$a$ + $b$" is two spans, not one wrapped answer
            if opener == "$" and _unescaped_dollars(inner):
                break
            if opener == "$$" and "$$" in inner:
                break
            if opener == "\\[" and ("\\[" in inner or "\\]" in inner):
                break
            s = inner.strip()
    return s


def extract_math_answer(text):
    """The final answer of a MATH generation, kept as the latex the model wrote.

    Same "the last label is the answer" rule as extract_answer -- a thinking trace
    restates the format spec before it commits -- but nothing is stripped that a
    mathematical answer could legitimately be made of.
    """
    if not text:
        return ""
    labels = list(re.finditer(r"answer\s*:", text, re.I))
    body = text[labels[-1].end():] if labels else text
    body = _MATH_CERTAINTY_LABEL.split(body, maxsplit=1)[0]   # the certainty closes it
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    ans = lines[0]
    # a display equation opened on the answer line runs onto the following ones
    for line in lines[1:]:
        if _latex_balanced(ans):
            break
        ans = f"{ans} {line}"
    ans = _MATH_CERTAINTY_SUFFIX.sub("", ans).strip()
    boxed = _last_boxed(ans)             # a model that boxes its answer means the box
    if boxed is not None:
        ans = boxed
    return _strip_math_delimiters(ans).strip().strip('"').strip()


def extract_confidence(text):
    """The certainty a generation reports, however it chose to format it."""
    if not text:
        return ""
    m = _CONFIDENCE_LABEL.search(text)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*:\s*(.*)", text, re.I | re.S)
    body = (m.group(1) if m else text).strip()
    lines = body.splitlines()
    if not lines:
        return ""
    m = re.search(r"[:\-]\s*(\d{1,3})\s*%?\s*$", lines[0])   # "Pulsar: 90"
    if m:
        return m.group(1)
    for line in lines[1:]:                                   # "Ruddigore\n70"
        m = re.fullmatch(r"\s*(\d{1,3})\s*%?\s*", line)
        if m:
            return m.group(1)
    return ""


def compute_evals(model_answer, ground_truth_answers, is_math=False):
    if is_math:
        # latex, not prose: lower-casing and clean_answer's punctuation surgery both
        # corrupt it, and \frac{1}{2} has to be compared as a number rather than as a
        # string, so token overlap says nothing and f1 is the match itself
        model_answer = extract_math_answer(model_answer)
        exact_match = bool(metric_max_over_ground_truths(
            math_exact_match_score, model_answer, ground_truth_answers))
        return {
            "exact_match": exact_match,
            "f1": float(exact_match),
            "ground_truth_answers": ground_truth_answers,
        }
    model_answer = extract_answer(model_answer).lower()
    model_answer = clean_answer(model_answer)
    exact_match = metric_max_over_ground_truths(exact_match_score, model_answer, ground_truth_answers)
    f1 = metric_max_over_ground_truths(f1_score, model_answer, ground_truth_answers)
    return {
        "exact_match": exact_match,
        "f1": f1,
        # "bertscore": bertscore_f1,
        "ground_truth_answers": ground_truth_answers
    }

def clean_answer(answer):
    return answer.split(".")[0].replace("is ", "").replace("should be ", "").replace(".", "").strip()

def safe_log(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps))

def compute_attn_eig_prod(attentions, layer_num=20, tok_lens=[], use_toklens=True):
    eigscore = torch.tensor(0.0, device=attentions.device)
    layer_num = attentions.shape[0]
    layer_results = []
    for layer in range(layer_num):
        # !!! uncomment below for original score. We found that this score performs better or similar.
        # eigscore = torch.tensor(0.0, device=attentions.device)
        for attn_head_num in range(len(attentions[layer])):  # iterating over number of attn heads
            # attns[i][layer_num][j] is of size seq_len x seq_len
            Sigma = attentions[layer][attn_head_num]
            if use_toklens and tok_lens:
                i1, i2 = tok_lens[0], tok_lens[1]
                Sigma = Sigma[i1:i2, i1:i2]
            Sigma = Sigma.to(torch.float32)
            # avoid -inf with clamp
            eigscore += safe_log(torch.diagonal(Sigma, 0)).mean()
        layer_results.append(eigscore.item())
    return layer_results


def compute_hidden_states(hidden_states, token_idx):
    # Sort token_idx by values
    token_idx = dict(sorted(token_idx.items(), key=lambda x: x[1]))
    chunk_ends = list(token_idx.values())
    chunk_keys = list(token_idx.keys())
    return_dict = {}
    # chunk up the entropy
    hidden_states = hidden_states.squeeze(0) # [T, hidden_size]
    start = 0
    for i in range(len(chunk_ends)):
        end = chunk_ends[i]
        return_dict[f"embedding_{chunk_keys[i]}"] = hidden_states[start:end, :].mean(dim=0).flatten().tolist()
        start = end
    return return_dict

def compute_icr(attentions, hidden_states, token_idx: dict, inputs: dict, top_k: int = 20):
    # ICR Probe feature (arXiv:2507.16488): token-averaged ICR score per layer.

    device = hidden_states.device
    # ipdb.trace()
    if device.type != "cuda" or attentions.device != device:
        raise RuntimeError(
            f"compute_icr expects hidden_states and attentions on the same CUDA device, "
            f"got hidden_states={device}, attentions={attentions.device}"
        )

    L, H, T, _ = attentions.shape
    input_lens = token_idx["question_end"]        # prompt/context length
    if input_lens >= T:                           # nothing was generated
        return {"icr_score": [None] * L}

    # # [step][layer] with a batch dim: step 0 = prompt, steps 1.. = one per new token
    # hs_steps = [[hidden_states[l, :input_lens].unsqueeze(0) for l in range(L + 1)]]
    # hs_steps += [[hidden_states[l, p].view(1, 1, -1) for l in range(L + 1)]
    #              for p in range(input_lens, T)]
    # attn_steps = [[attentions[l, :, :input_lens, :input_lens].unsqueeze(0) for l in range(L)]]
    # attn_steps += [[attentions[l, :, p, :p + 1].unsqueeze(0).unsqueeze(2) for l in range(L)]
    #                for p in range(input_lens, T)]

    core_positions = {
        "user_prompt_start": token_idx.get("template_end", 0),
        "user_prompt_end": input_lens,
        "response_start": input_lens,
    }
    scorer = ICRScore(hidden_states, attentions, core_positions=core_positions,
                      icr_device=device)
    icr_by_layer, _ = scorer.compute_icr(top_k=top_k, top_p=None, pooling="mean",
                                         attention_uniform=False, hidden_uniform=False,
                                         use_induction_head=False)
    # [layer][token] -> mean over answer tokens -> [L]; mirrors lookback_lens_ratio
    icr_score = [sum(tok) / len(tok) if tok else None for tok in icr_by_layer]
    return {"icr_score": icr_score}

def process_single_output(
    hf_out: Mapping[str, Any],
    vllm_out: Mapping[str, Any],
    processor: Any,
    all_token_ids: bool = False,
    idx: Optional[int] = None,
    is_math: bool = False,
) -> Dict[str, Any]:

    # --- Normalize token ids
    answer_ids = torch.as_tensor(vllm_out.get("answer_token_ids"), dtype=torch.long)
    prompt_ids = torch.as_tensor(vllm_out.get("prompt_token_ids"), dtype=torch.long)
    len_answer = int(answer_ids.numel())
    len_prompt = int(prompt_ids.numel())
    seq_len = len_answer + len_prompt

    logits = hf_out["logits"]                     # [T, V] or [B, T, V]
    attentions = hf_out["attentions"]             # [..., T, T]
    hidden_states = hf_out.get("hidden_states")   # model-dependent
    hidden_states_all = hf_out.get("hidden_states_all")  # [L+1, T, D] all layers, for ICR
    # attention_mask is currently unused, but kept for future logic:
    _ = hf_out.get("attention_mask")

    # Squeeze batch dim for logits if present
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)

    if logits.ndim == 2:
        t_logits = logits.shape[0]               # [T, V]
    elif logits.ndim == 3:
        t_logits = logits.shape[1]               # [B, T, V]
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    if t_logits != seq_len:
        # raise ValueError(f"Logits time dim {t_logits} != prompt+answer {seq_len}")
        print(f"Warning: Logits time dim {t_logits} != prompt {len_prompt}+answer {len_answer}, continuing...")
        return None

    if attentions is not None:
        if attentions.shape[-1] != seq_len or attentions.shape[-2] != seq_len:
            raise ValueError(
                f"Attentions expected square [...,{seq_len},{seq_len}], got {tuple(attentions.shape)}"
            )

    logprobs = compute_logprobs(logits, processor)

    # Window [start, end] for eig-score; defaults to the end of the answer
    start, end = 0, max(0, len_answer - 1)
    token_idx = {"answer_end": end}

    if not all_token_ids:
        # Full sequence ids for tokenizer-aware indexing
        full_ids = (prompt_ids.tolist() + answer_ids.tolist())

        if "qwen" in processor.name_or_path.lower():
            ti = compute_token_idx_qwen(full_ids, processor)  # {name: position}
        elif "llama" in processor.name_or_path.lower():
            ti = compute_token_idx_llama(full_ids, processor)
        elif "olmo" in processor.name_or_path.lower():
            ti = compute_token_idx_olmo(full_ids, processor)
        elif "gemma-4" in processor.name_or_path.lower():
            ti = compute_token_idx_gemma4(full_ids, processor)
        else:
            raise ValueError("Processor not recognized for token idx computation.")
        
        ti = dict(sorted(ti.items(), key=lambda kv: kv[1]))  # sort by position
        token_idx = ti

        if "answer_end" in ti:
            keys = list(ti)
            a_pos = keys.index("answer_end")
            start = list(ti.values())[a_pos - 1] if a_pos > 0 else 0
            end = ti["answer_end"]

    # Attention/hidden-state features
    # entropy_vectors = compute_entropy(attentions, token_idx, vllm_out)
    # renyi_vectors = compute_renyi(attentions, token_idx, vllm_out)
    hidden_states_vectors = compute_hidden_states(hidden_states, token_idx)
    lookback_lens_vectors = compute_lookback_lens(attentions, token_idx)
    trace_jacobian_vectors = compute_trace_jacobian(attentions, token_idx, vllm_out )
    icr_vectors = compute_icr(attentions, hidden_states_all, token_idx, vllm_out)

    # Keep layer_num <= available layers when possible
    layer_num = 20
    if isinstance(attentions, torch.Tensor) and attentions.ndim >= 3:
        maybe_layers = attentions.shape[0] if attentions.ndim in (4, 5) else None
        if maybe_layers:
            layer_num = min(layer_num, maybe_layers)

    eigscore = compute_attn_eig_prod(attentions, layer_num=layer_num,
                                     tok_lens=[start, end], use_toklens=True)

    # --- Evals (robust to missing keys)
    evals: Dict[str, Any] = {}
    if "model_answer" in vllm_out and "ground_truth_answers" in vllm_out:
        gts = vllm_out.get("ground_truth_answers") or []
        if len(gts) == 0:
            raise ValueError("ground_truth_answers is empty.")
        evals = compute_evals(vllm_out["model_answer"], gts, is_math=is_math)
    elif "exact_match" in vllm_out:
        evals = {"exact_match": vllm_out["exact_match"]}

    # --- ID & message
    # prefer explicit idx, then question_id, then id
    question_id = idx if idx is not None else vllm_out.get("question_id", vllm_out.get("id"))

    # message/output payload: try "output" → "model_answer" → "answer"
    message = vllm_out.get("output")
    if message is None:
        message = vllm_out.get("model_answer") or vllm_out.get("answer") or {}

    # --- Build lightweight vllm_out (remove large arrays)
    vllm_out_clean = dict(vllm_out)  # shallow copy
    vllm_out_clean.pop("prompt_token_ids", None)
    vllm_out_clean.pop("answer_token_ids", None)
    vllm_out_clean.pop("output", None)

    return {
        "output": {
            "id": question_id,
            "logprobs": logprobs,
            "usage": {
                "completion_tokens": len_answer,
                "prompt_tokens": len_prompt,
                "total_tokens": seq_len,
            },
            "message": message,
        },
        "lookback_lens_ratio" : lookback_lens_vectors,
        "attn_eig_prod": eigscore,
        **icr_vectors,
        **vllm_out_clean,
        # **entropy_vectors,
        **hidden_states_vectors,
        **trace_jacobian_vectors,
        # **renyi_vectors,
        **token_idx,
        **evals,
    }

def scan_processed(out_dir, timestemp=None):
    """Absolute indices already written to out_dir, and the highest file counter used.

    Records carry their input index at output.id (process_single_output stores idx there),
    so a resumed run can skip exactly what is already on disk. The counter is needed so the
    resumed run writes new files instead of overwriting the last ones.
    """
    done, max_count = set(), 0
    # any timestamp, not only this run's: every invocation gets its own, and a resumed
    # run must continue from the files the earlier invocations wrote
    pattern = os.path.join(out_dir, "results_*_*.json")
    for path in glob.glob(pattern):
        if "vllm" in os.path.basename(path) or "shap" in os.path.basename(path):
            continue
        try:
            max_count = max(max_count, int(path.rsplit("_", 1)[-1].removesuffix(".json")))
        except ValueError:
            continue
        try:
            with open(path) as f:
                records = json.load(f)
        except Exception as e:
            print(f"Could not read {path} while scanning for resume: {e}")
            continue
        for rec in records if isinstance(records, list) else [records]:
            idx = (rec.get("output") or {}).get("id")
            if isinstance(idx, int):
                done.add(idx)
    return done, max_count


def send_to_hf(
    model,
    processor,
    generated_data,
    timestemp,
    all_token_ids=False,
    out_dir=None,
    start_idx=0,
    skip_idxs=None,
    count_start=0,
    is_math=False,
):
    count = count_start
    skip_idxs = skip_idxs or set()
    if not torch.cuda.is_available():
        raise RuntimeError("send_to_hf requires a CUDA device: feature extraction runs on GPU")
    # model.device is unreliable under device_map="auto" (reports cpu when the first
    # block is offloaded), so pin the feature tensors to an explicit device.
    device = torch.device("cuda")
    pad_id = processor.pad_token_id
    if pad_id == None:
        pad_id = processor.eos_token_id
    assert pad_id is not None, "pad_id must be set in processor"
    records = []
    count_errors = 0
    # output_attentions materialises a [layers, heads, T, T] map, so the memory a single
    # example needs grows with the square of its length: on a 64-layer/64-head model a
    # 2k-token sequence alone asks for ~38 GiB, and the stack below wants a second copy.
    # Measure the budget once and skip the few outliers, rather than losing the run.
    # Gemma-4 ships a unified multimodal config that nests the text tower's dims under
    # text_config (48 layers x 16 heads); Llama, Qwen and OLMo expose them at the top
    # level, where the getattr falls through to the config itself and nothing changes.
    text_config = getattr(model.config, "text_config", model.config)
    heads_x_layers = text_config.num_hidden_layers * text_config.num_attention_heads
    skipped_long = []
    # generated_data is already sliced from start_idx, so the absolute index of a local
    # position is start_idx + local -- indexing generated_data[i] would offset it twice.
    for local, item in enumerate(tqdm(generated_data)):
        i = start_idx + local
        if i in skip_idxs:
            continue

        seq_tensor = torch.tensor(
            item["prompt_token_ids"] + item["answer_token_ids"],
            dtype=torch.long
        )
        try:
            full_token_ids = seq_tensor.unsqueeze(0).to(model.device)
            attention_mask = (full_token_ids != pad_id).long().to(model.device)    
            with torch.inference_mode():
                gen_out = model(input_ids=full_token_ids, attention_mask=attention_mask, output_attentions=True, output_scores=True, output_hidden_states=True)
        except Exception as e:
            print(f"Error processing item {i}, skipping...: {e}")
            continue
        logits = gen_out["logits"]
        assert logits.shape[1] == seq_tensor.shape[0], f"Expected logits length {seq_tensor.shape[1]}, got {logits.shape[1]}"

        try:
            attentions = torch.stack(gen_out["attentions"], dim=0)# [layers, batch_size, heads, T, T]
            gen_out["attentions"] = None
        except:
            try:
                # [layers, batch_size, heads, T, T]
                att = gen_out["attentions"]
                L = len(att)
                B, H, T, _ = att[0].shape
                attentions = torch.empty((L, B, H, T, T), dtype=torch.float16)  # CPU tensor
                for idx_t, t in enumerate(att):
                    attentions[idx_t].copy_(t.to(dtype=torch.float16))
                gen_out["attentions"] = None
            except Exception as e:
                print(f"Error processing attentions for item {i}, skipping...: {e}")
                count_errors += 1
                continue


        # [layers, batch_size, T, hidden_size]
        # hidden_s= gen_out["hidden_states"]
        # ipdb.set_trace()
        # hidden_states = torch.empty((len(hidden_s),  hidden_s[0].shape[0], hidden_s[0].shape[1], hidden_s[0].shape[2]), dtype=torch.float16)  # CPU tensor
        # for idx_h, h in enumerate(hidden_s):
        #     hidden_states[idx_h].copy_(h.to(dtype=torch.float16))
        hidden_states = torch.stack([h.to(torch.float16) for h in gen_out["hidden_states"]])
        gen_out["hidden_states"] = None
        # [layers, batch_size, T, hidden_size] = torch.stack(hidden_states, dim=0) if hidden_states is not None else None
        record = process_single_output(
            hf_out={
                "logits": logits[0],
                "attentions": attentions[:, -1].to(device),
                "attention_mask": attention_mask[0],
                "hidden_states": hidden_states[-1, -1].to(device), # last layer, last batch item
                "hidden_states_all": hidden_states[:, -1].to(device), # all layers, last batch item [L+1, T, D]
            },
            vllm_out=item,
            processor=processor,
            all_token_ids=all_token_ids,
            idx=i,
            is_math=is_math,
        )
        if record==None:
            print(f"Skipping record {i} because it is None")
            continue
        records.append(record)
        count += 1 # counts over path_to_vllm_out files continuously
        del logits, attentions, attention_mask, gen_out, seq_tensor, full_token_ids, hidden_states
        # torch.cuda.empty_cache() 
        if count % 5 == 0:
            path_to_out_hf = f"{out_dir}/results_{timestemp}_{count}.json"
            with open(path_to_out_hf, "w") as f:
                json.dump(records, f, indent=2)
            del records
            # torch.cuda.empty_cache()
            records = []

    # flush the tail: without this the final <5 records of every run are lost, which on a
    # resumed run means losing up to 4 records at each stop
    if records:
        path_to_out_hf = f"{out_dir}/results_{timestemp}_{count}.json"
        with open(path_to_out_hf, "w") as f:
            json.dump(records, f, indent=2)

    print(f"Processed {count - count_start} records with {count_errors} errors")
    if skipped_long:
        print(f"Skipped {len(skipped_long)} items that did not fit in memory: {skipped_long}")
    return
