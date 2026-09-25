import ipdb
import re
import numpy as np
from run.utils.triviaqa_evaluation import exact_match_score, f1_score, metric_max_over_ground_truths, normalize_answer
from run.utils.math_evaluation import math_exact_match_score
from run.utils.math_labels import gold_answers
from run.utils.utils_generation import extract_math_answer
from tqdm import tqdm
import torch
from sklearn.isotonic import IsotonicRegression
import json 
from sklearn.metrics import average_precision_score, roc_auc_score
import math
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

def exclude_list(data, exclude_idxs):
    # a set, not the list handed in: excluding truncated generations puts thousands of
    # indices in here, and a list membership test would make this quadratic
    exclude_idxs = set(exclude_idxs)
    return [x for i, x in enumerate(data) if i not in exclude_idxs]

def _load(fp):
    with open(fp) as f:
        return json.load(f)


# def bertscore_adapter(prediction: str, ground_truth: str, **kwargs) -> float:
#     P, R, F1 = bertscore([prediction], [ground_truth], **kwargs)
#     return float(F1[0].item())  # get scalar

def correct_data(raw_data, math=False):
    """Re-score every record, reading MATH the way MATH has to be read.

    ``math=True`` switches two things that are wrong for latex answers: the reference,
    which the runs on disk store as "No answer" for 42% of the items (see
    run.utils.math_labels), and the comparison, which has to be mathematical rather
    than string equality.
    """
    count = 0
    unmatched = 0
    corrected_data = []
    for i in tqdm(raw_data):
        answer = i["model_answer"]
        try:
            output = i["output"]["message"] if "message" in i["output"].keys() else i["output"]
        except:
            pass
        if answer == "":
            try:
                if "</think>\n\nAnswer: " in output and "\nCertainty: " in output:
                    answer = output.split("</think>\n\nAnswer: ")[-1].split("\nCertainty: ")[0]
                elif "\nCertainty: " in output:
                    answer = output.split("\nCertainty: ")[0].replace("Answer: ", "").strip()
                elif ": " in output:
                    answer = output.split(": ")[0].strip()
                else:
                    raise ValueError(f"Unknown format for answer: {output}")
            except:
                answer = i["model_answer"]
        if math:
            # re-read from the generation rather than trust what is on disk: the runs
            # were written by an extractor that read "-2" as a bare confidence and
            # stored "" for it, which emptied 3.8% of Llama-3.2-3B's MATH answers.
            # normalize_answer() is skipped as well -- it strips every punctuation
            # mark, turning \frac{9}{7} into "frac 9 7" and \sqrt{2} into "sqrt 2"
            answer = extract_math_answer(output) or str(answer).strip()
        else:
            answer = normalize_answer(answer).replace(".", "").replace("*", "")
        certainty = ""
        if answer != "" and "\nCertainty: " in output:
            certainty = output.split("\nCertainty: ")[-1]
        elif answer != "" and ": " in output:
            certainty = output.split(": ")[-1]

        if answer == "":
            answer = output
        count +=1
        if math:
            # the reference travelling with the record is unusable for 42% of MATH, so
            # it is read from the dataset again rather than scored against
            ground_truth_answers = gold_answers(i["question"], i.get("question_id"))
            if ground_truth_answers is None:
                unmatched += 1
                ground_truth_answers = i["ground_truth_answers"]
            i["ground_truth_answers"] = ground_truth_answers
            exact_match = bool(metric_max_over_ground_truths(
                math_exact_match_score, answer, ground_truth_answers))
            # token overlap says nothing about whether two latex answers agree
            f1 = float(exact_match)
        else:
            ground_truth_answers = i["ground_truth_answers"]
            exact_match = metric_max_over_ground_truths(exact_match_score, answer, ground_truth_answers)
            f1 = metric_max_over_ground_truths(f1_score, answer, ground_truth_answers)
        i["exact_match"] = exact_match
        assert isinstance(i["exact_match"], bool)
        i["f1"] = f1
        # i["bertscore_f1"] = bertscore_f1
        confidence = safe_str_to_float(certainty)
        # the fallback above can hand a whole answer to float(): one MATH generation
        # ending in 60 ones overflowed to inf, and a single inf makes the mean that
        # fills in every missing certainty inf as well
        i["model_confidence"] = confidence if np.isfinite(confidence) else np.nan
        i["model_answer"] = answer
        corrected_data.append(i)

    # fill up model confidence with the average of the non-nan values
    model_confidence = np.array([i["model_confidence"] for i in corrected_data])
    model_confidence_mean = np.nanmean(model_confidence)
    model_confidence[np.isnan(model_confidence)] = model_confidence_mean

    # make 0 if all nan
    if np.all(np.isnan(model_confidence)):
        model_confidence_mean = 0.0
        model_confidence[np.isnan(model_confidence)] = 0.0

    corrected_data_mean_certainty = []
    for i, item in enumerate(corrected_data):
        item["model_confidence"] = model_confidence[i]
        corrected_data_mean_certainty.append(item)

    print(f"Number of correct data: {count}")
    if math and unmatched:
        print(f"WARNING: {unmatched}/{count} MATH problems were not found in the dataset "
              f"and kept the reference stored with the run")
    # print("Check the distrubution of the model confidence with a histogram")
    # plt.hist(model_confidence, bins=100)
    # plt.savefig("model_confidence_histogram.png")
    # plt.close()
    return corrected_data_mean_certainty


def to_prob(x, y_ref=None, kind="prob"):
    if kind == "prob":      return x.float() #.clamp_(0, 1)
    if kind == "log_prob":  return x.float().exp()
    if kind == "logit":     return x.float().sigmoid()
    if kind == "calibrate":
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            x.cpu().numpy(), y_ref.cpu().numpy())
        return torch.as_tensor(iso.transform(x.cpu().numpy()))
    raise ValueError(kind)


def safe_str_to_float(s):
    if isinstance(s, str):
        s = s.replace("%", "").replace(":", "").replace("percent", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan
    

def compute_token_lengths(data, section="answer_end"):
    start_section = None
    end_section = section
    if section == "answer_end":
        start_section = "question_end"
    elif section == "question_end":
        start_section = "template_end"
    elif section == "think_chunk":
        start_section = "think_chunk_1"
        end_section = "answer_end"
    token_lengths = []
    for item in tqdm(data):
        if section == "answer_end" and "think_chunk_1" in item.keys():
            # find last think chunk
            start_section = [k for k in item.keys() if k.startswith("think_chunk_")]
            start_section = max(start_section, key=lambda k: int(re.search(r"\d+$", k).group()))
        try:
            token_length = item[end_section] - item[start_section]
            token_lengths.append(torch.tensor(token_length, dtype=torch.float))
        except Exception:   
            ipdb.set_trace()
            continue
    average_token_length = torch.stack(token_lengths, dim=0).mean()
    return average_token_length

def compute_all_token_lengths(data):
    lengths = {}
    sections = ["question_end", "answer_end"]
    if "think_chunk_1" in data[0].keys():
        sections.append("think_chunk")
    for section in sections:
        lengths[section] = compute_token_lengths(data, section=section).item()
        print(f"Average token length for {section}: {lengths[section]:.2f}")
    return lengths


def get_lengths_questions_answers(data_val, skip_idxs=None):
    assert isinstance(data_val, list) and len(data_val) > 0 and isinstance(data_val[0], dict), "data_val must be a list of dicts"
    # check if data_val has think   
    return [item["answer_end"] for item in data_val if skip_idxs is None or data_val.index(item) not in skip_idxs]
   

def bootstrap_metric(metric_fn, y_true, y_score, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    # y_true = y_true.numpy()
    # y_score = y_score.numpy()
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            score = metric_fn(y_true[indices], y_score[indices])
        except:
            continue
        scores.append(score)
    scores = np.array(scores)
    mean = np.mean(scores)
    ci_low = np.percentile(scores, 2.5)
    ci_high = np.percentile(scores, 97.5)
    return mean, (ci_low, ci_high)



def get_results(reference: torch.Tensor, score: torch.Tensor, reference_train=None, score_train=None, kind="prob", n_bootstraps=1000, compute_ci=False):
    reference, score = map(torch.as_tensor, (reference, score))
    if reference_train is not None and score_train is not None:
        reference_train, score_train = map(torch.as_tensor, (reference_train, score_train))
    else:
        reference_train = score_train = None

    p_test  = to_prob(score, reference, kind)
    p_train = to_prob(score_train, reference_train, kind) if score_train is not None else None
    stats = dict(
        overall_test = float(reference.float().mean()),

        pr_auc_test  = average_precision_score(reference, p_test),
        roc_auc_test = roc_auc_score(reference, p_test),
        roc_auc_train = roc_auc_score(reference_train, p_train) if p_train is not None else np.nan,
        roc_auc_train_ci_low = np.nan,
        roc_auc_train_ci_high = np.nan,
        pr_auc_train  = average_precision_score(reference_train, p_train) if p_train is not None else np.nan,
    )

    if compute_ci:
        # the raw AUC, the same quantity as roc_auc_test above. Bootstrapping
        # max(auc, 1 - auc) instead put the interval on the mirror of the estimate
        # whenever a feature pointed the wrong way, so attn_score on hotpot was
        # reported as 0.291 with a CI of [0.693, 0.723]
        _, roc_auc_ci = bootstrap_metric(
            roc_auc_score, reference, p_test, n_bootstraps
        )
        stats.update(
            roc_auc_test_ci_low = roc_auc_ci[0],
            roc_auc_test_ci_high = roc_auc_ci[1],
        )

    print(f"Overall Test Prevalence: {stats['overall_test']:.4f}")
    print(stats)
    return stats













def get_data(prefix, max_workers=None, subset=None):
    files = [p for p in glob.glob(f"{prefix}/results_20*.json") if "vllm" not in p and "shap" not in p]
    files.sort(key=lambda p: int(p.rsplit("_",1)[-1].removesuffix(".json")))
    if subset: files = files[:min(subset, len(files))]

    if len(files) == 0:
        print(f"No files found in {prefix}")
        return []

    max_workers = 4

    data = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_load, fp): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Parsing", unit="file"):
            fp = futs[fut]
            try:
                item = fut.result()
                data.extend(item if isinstance(item, list) else [item])
            except Exception as e:
                print(f"Error {fp}: {e}")

    return data

def truncation_budget(data, min_share=0.01):
    lengths = []
    for item in data:
        try:
            lengths.append(int(item["output"]["usage"]["completion_tokens"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not lengths:
        return None
    longest = max(lengths)
    if lengths.count(longest) < max(2, min_share * len(lengths)):
        return None
    return longest


def get_entropy(data, section="trace_jacobian_answer_end", create_figures=True, correct=True, label_metric="exact_match", normalize=True, skip_truncated=False, math=False):
    labels_list: list[torch.Tensor] = []
    entropies_list: list[torch.Tensor] = []
    f1_list: list[torch.Tensor] = []
    print("Using metric for labels: ", label_metric)
    if correct:
        print("Correcting data")
        data_corrected = correct_data(data, math=math)
    else:
        print("Not correcting data")
        data_corrected = data

    budget = truncation_budget(data_corrected) if skip_truncated else None
    if skip_truncated:
        print(f"Truncation budget detected: {budget}"
              if budget else "No truncated generations in this run")

    exclude_idxs = []
    for item in tqdm(enumerate(data_corrected)):
        idx, item = item
        try:
            assert section in item, f"Section {section} not in item keys {item.keys()} for index {idx}"
            if budget is not None:
                # a generation that ran out of room never reached its answer, so its
                # label says the budget was too small, not that the model was wrong
                assert item["output"]["usage"]["completion_tokens"] < budget, \
                    f"Skipping truncated generation at index {idx}"
            entropy = torch.tensor(item[section], dtype=torch.float32)
            entropies_list.append(entropy)

            if label_metric == "f1":
                labels_list.append(torch.tensor(item["f1"] > 0.5))
            elif label_metric == "bertscore_f1":
                labels_list.append(torch.tensor(item["bertscore_f1"] > 0.5))
            else:
                labels_list.append(torch.tensor(item["exact_match"]))

            f1_list.append(torch.tensor(item["f1"]))

        except Exception:
            exclude_idxs.append(idx)
            continue
    print(f"Excluded {len(exclude_idxs)} examples due to missing or invalid {section} data.")
    
    if len(entropies_list) == 0:
        ipdb.set_trace()
    entropies = torch.stack(entropies_list, dim=0)  # [N, ...]
    labels = torch.stack(labels_list, dim=0)        # [N]
    f1_scores = torch.stack(f1_list, dim=0)         # [N]
    return entropies, labels, f1_scores, exclude_idxs

def get_token_probs(data, skip_idxs=None):
    token_probs = []
    for item in tqdm(data):
        answer_start = item["question_end"]
        if "think_chunk_1" in item.keys():
            think_chunks = {k: v for k, v in item.items() if k.startswith("think_chunk_")}
            last_key = max(think_chunks, key=lambda k: int(k.split("_")[-1]))
            answer_start = think_chunks[last_key]
        try:
            logprob_list = item["output"]["logprobs"][answer_start:item["answer_end"]]
            logprobs = torch.exp(torch.tensor([i["logprob"] for i in logprob_list])).nanmean()
            if logprobs is None or not math.isfinite(logprobs):
                probs = torch.tensor(float('nan'))
            else:
                probs = torch.tensor(logprobs, dtype=torch.float)
        except (KeyError, IndexError, TypeError):
            probs = torch.tensor(float('nan'))
        token_probs.append(probs)

    token_probs = exclude_list(token_probs, skip_idxs) if skip_idxs is not None else token_probs
    token_probs = torch.stack(token_probs, dim=0)

    if torch.isnan(token_probs).any():
        mean_val = token_probs[~torch.isnan(token_probs)].mean()
        token_probs[torch.isnan(token_probs)] = mean_val

    assert isinstance(token_probs, torch.Tensor)
    return token_probs

def get_token_entropy(data, skip_idxs=None):
    token_probs = []
    for item in tqdm(data):
        answer_start = item["question_end"]
        if "think_chunk_1" in item.keys():
            think_chunks = {k: v for k, v in item.items() if k.startswith("think_chunk_")}
            last_key = max(think_chunks, key=lambda k: int(k.split("_")[-1]))
            answer_start = think_chunks[last_key]
        entropy_list = item["output"]["logprobs"][answer_start:item["answer_end"]]
        probs = torch.tensor([i["entropy"] for i in entropy_list]).mean(dim=0)
        token_probs.append(probs)
    token_probs = exclude_list(token_probs, skip_idxs) if skip_idxs is not None else token_probs

    token_probs = torch.stack(token_probs, dim=0) * -1
    token_probs[torch.isnan(token_probs)] = token_probs[~torch.isnan(token_probs)].mean()
    assert isinstance(token_probs, torch.Tensor)
    return token_probs

def get_model_confidence(data, skip_idxs=None):
    model_confidence = []
    for item in tqdm(data):
        conf = torch.tensor(safe_str_to_float(item["model_confidence"]))
        model_confidence.append(conf)
    model_confidence = exclude_list(model_confidence, skip_idxs) if skip_idxs is not None else model_confidence
    model_confidence = torch.stack(model_confidence, dim=0)
    model_confidence[torch.isnan(model_confidence)] = model_confidence[~torch.isnan(model_confidence)].mean()
    model_confidence[torch.isinf(model_confidence)] = model_confidence[~torch.isinf(model_confidence)].mean()

    model_confidence = model_confidence.clamp_(0, 100)
    assert isinstance(model_confidence, torch.Tensor)
    return model_confidence

def get_attn_score(data, skip_idxs=None):
    attn_scores = []
    errors = 0
    assert "attn_eig_prod" in data[0].keys(), "attn_eig_prod not in data keys"
    layer_len = len(data[0]["attn_eig_prod"])
    layer_idx = int(layer_len * 0.8) 
    for item in tqdm(data):
        try:
            # uncomment next line and comment out 425, for the original score. Performs worse or similar to the score we used. 
            # attn_scores.append(torch.tensor(item["attn_eig_prod"])[layer_idx] - torch.tensor(item["attn_eig_prod"])[layer_idx - 1])
            attn_scores.append(torch.tensor(item["attn_eig_prod"])[layer_idx])
        except:
            attn_scores.append(torch.tensor(0))
            errors += 1
    attn_scores = exclude_list(attn_scores, skip_idxs) if skip_idxs is not None else attn_scores
    attn_scores = torch.stack(attn_scores, dim=0)
    attn_scores[torch.isinf(attn_scores)] = attn_scores[~torch.isinf(attn_scores)].mean()

    print(f"Missed {errors} cases for attn_eig_prod")
    assert isinstance(attn_scores, torch.Tensor)
    return attn_scores

def zero_nan(vecs, name):
    nan_mask = torch.isnan(vecs)
    n_nan = int(nan_mask.sum())
    if n_nan:
        n_rows = int(nan_mask.reshape(nan_mask.shape[0], -1).any(dim=1).sum())
        print(f"Replaced {n_nan} NaN entries with 0 for {name} (affects {n_rows} records)")
        vecs = torch.nan_to_num(vecs, nan=0.0)
    return vecs


def get_hidden_state_score(data, semantic_section="embedding_answer_end", skip_idxs=None):
    vecs, errors = [], 0
    for item in data:
        try:
            v = torch.as_tensor(item[semantic_section], dtype=torch.float32)
        except Exception:          
            v, errors = torch.empty(0), errors + 1

        vecs.append(v)
    vecs = exclude_list(vecs, skip_idxs) if skip_idxs is not None else vecs
    vecs = torch.stack(vecs)
    vecs = zero_nan(vecs, "hidden")
    print(f"Missed {errors} cases for hidden")
    assert isinstance(vecs, torch.Tensor)
    return vecs

def get_lr(data, skip_idxs=None):
    vecs, errors = [], 0
    for item in data:
        try:
            v = torch.as_tensor(item["lookback_lens_ratio"], dtype=torch.float32)
        except Exception:          
            v, errors = torch.empty(0), errors + 1

        vecs.append(v)
    vecs = exclude_list(vecs, skip_idxs) if skip_idxs is not None else vecs
    vecs = torch.stack(vecs)
    vecs = zero_nan(vecs, "lookback_lens_ratio")
    print(f"Missed {errors} cases for hidden")
    assert isinstance(vecs, torch.Tensor)
    return vecs

def get_icr(data, skip_idxs=None):
    vecs, errors = [], 0
    for item in data:
        try:
            v = torch.as_tensor(item["icr_score"], dtype=torch.float32)
        except Exception:
            v, errors = torch.empty(0), errors + 1
        vecs.append(v)
    vecs = exclude_list(vecs, skip_idxs) if skip_idxs is not None else vecs
    vecs = torch.stack(vecs)
    vecs = zero_nan(vecs, "icr")
    print(f"Missed {errors} cases for icr")
    assert isinstance(vecs, torch.Tensor)
    return vecs
