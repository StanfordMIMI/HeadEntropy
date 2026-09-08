import argparse
import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMo-2-0425-1B"
ROOT = "/local/home/osophie/HeadEntropy/results_vqa"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_scores")

# (revision, glob for the generation run that supplies prompts/gold answers)
STAGES = [
    ("stage1-step480000-tokens1007B",        "*train_stage1-step480000-*"),
    ("stage1-step1020000-tokens2140B",       "*train_stage1-step1020000-*"),
    ("stage1-step1840000-tokens3859B",       "*train_stage1-step1840000-*"),
    ("stage2-ingredient1-step23852-tokens51B", "*train_stage2-ingredient1-*"),
    ("stage2-ingredient2-step23852-tokens51B", "*train_stage2-ingredient2-*"),
]


def read_examples(pattern, limit=None):
    """(question_id, prompt, gold) from a generation run, in file order."""
    d = sorted(glob.glob(os.path.join(ROOT, pattern)))[-1]
    files = [p for p in glob.glob(f"{d}/results_*.json") if "vllm" not in p]
    files.sort(key=lambda p: int(p.rsplit("_", 1)[-1].removesuffix(".json")))
    out = []
    for f in files:
        for it in json.load(open(f)):
            gts = [g for g in (it.get("ground_truth_answers") or []) if g and g.strip()]
            if not gts:
                continue
            # the first alias, consistently across checkpoints. Which alias is used
            # shifts the level of logp but not its change between checkpoints, which
            # is what the grouping below is built from.
            out.append((it["question_id"], it["prompt"], gts[0]))
            if limit and len(out) >= limit:
                return out
    return out


@torch.no_grad()
def score(revision, examples, device="cuda"):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, torch_dtype=torch.float32,
        attn_implementation="eager",            # output_attentions needs eager
    ).to(device).eval()

    qids, lp_sum, lp_mean, tjs = [], [], [], []
    for qid, prompt, gold in tqdm(examples, desc=revision[:28], unit="ex"):
        p_ids = tok(prompt, return_tensors="pt").input_ids
        g_ids = tok(" " + gold.strip(), add_special_tokens=False, return_tensors="pt").input_ids
        if g_ids.shape[1] == 0:
            continue
        ids = torch.cat([p_ids, g_ids], dim=1).to(device)
        n_p, n_g = p_ids.shape[1], g_ids.shape[1]

        out = model(ids, output_attentions=True)
        # log P of each gold token, predicted from the position before it
        logits = out.logits[0, n_p - 1:-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(1, ids[0, n_p:].unsqueeze(1)).squeeze(1)

        # per-head trace-Jacobian over the gold answer rows, 1 - sum_j p^2,
        # matching row_trace_jacobian in run/utils/utils_generation.py
        att = torch.stack([a[0] for a in out.attentions])          # [L, H, T, T]
        rows = att[:, :, n_p:, :].float()
        tj = (1 - (rows ** 2).sum(-1)).mean(-1)                     # [L, H]

        qids.append(qid); lp_sum.append(float(lp.sum())); lp_mean.append(float(lp.mean()))
        tjs.append(tj.flatten().cpu().numpy())
        del out, att, rows

    del model
    torch.cuda.empty_cache()
    return (np.array(qids), np.array(lp_sum), np.array(lp_mean), np.stack(tjs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="examples per checkpoint")
    ap.add_argument("--only", type=str, default=None, help="substring of one revision")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for revision, pattern in STAGES:
        if a.only and a.only not in revision:
            continue
        ex = read_examples(pattern, a.limit)
        qids, s, m, tj = score(revision, ex)
        path = os.path.join(OUT, f"{revision}.npz")
        np.savez_compressed(path, qids=qids, logp_sum=s, logp_mean=m, tj=tj)
        print(f"{revision}: {len(qids)} examples, tj {tj.shape}, "
              f"logp/token mean {m.mean():.3f} -> {path}")
