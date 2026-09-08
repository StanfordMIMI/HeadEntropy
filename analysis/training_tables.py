"""Regenerate the two OLMo pretraining tables in the paper (tab:flips and
tab:level-change, sec/4_method.tex) from the raw checkpoint runs.

Neither table had generating code: the numbers lived only in the .tex. This
rebuilds both from results_vqa, so they can be re-derived when the subset, the
entropy definition or the checkpoint list changes.

Use from a notebook:

    from training_tables import load, table_flips, table_level_change
    H, Y, qids = load()                 # H[stage, example, head], Y[stage, example]
    flips, wil = table_flips(H, Y)
    lvl        = table_level_change(H, Y)

or run it: ``python analysis/training_tables.py`` prints both frames and the
LaTeX for each.

Known discrepancy: this does not reproduce the numbers in the .tex. The signs do
(wrong->right falls, right->wrong rises, 0->1 inverts) but every magnitude and
count differs, and the checkpoint list is the likely reason.

Two things the published counts pin down without knowing the sample size. First,
accuracy is flat: the two flip directions balance at every transition (86/83,
79/84, 86/84) and stayed-wrong barely moves (585, 589, 587). Second, ~22% of
examples change correctness per transition. Flat accuracy with that much churn is
the signature of sampling noise rather than learning. The decisive mismatch is the
stayed-wrong row -- the paper has |dH| < 0.002, no global drift, which is what
licenses reading the flip groups as a real effect; the local runs drift -0.28.

The local temp0.0 checkpoints do not behave that way: exact-match climbs
monotonically (0.0, 0.3, 1.6, 2.3, 1.8, 3.5, 8.7, 17.4, 22.7, 23.7%), is never
flat across four consecutive checkpoints, and flips 10-16% of examples. The
generating code was never committed, and the runs it used are most likely the
temp1.3 twins referenced by analyses/training_analysis.ipynb in the old repo,
which are not mirrored here. Set STAGES/N_FILES/ENTROPY_KEYS to whatever the paper
actually used and the tables below follow.
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = "/local/home/osophie/HeadEntropy/results_vqa"

# The five OLMo-2-1B checkpoints, in training order -- indices 0..4 as the tables
# name them. tab:level-change covers all four transitions; tab:flips drops 0->1,
# where every head collapses out of random initialisation at once and swamps the
# per-example effect. Matched by suffix because the timestamp prefix differs per
# run and the temperature in the directory name may not.
STAGES = [
    ("step480k",  "*OLMo-2-0425-1B_train_stage1-step480000-*"),
    ("step1020k", "*OLMo-2-0425-1B_train_stage1-step1020000-*"),
    ("step1840k", "*OLMo-2-0425-1B_train_stage1-step1840000-*"),
    ("stage2-ing1", "*OLMo-2-0425-1B_train_stage2-ingredient1-*"),
    ("stage2-ing2", "*OLMo-2-0425-1B_train_stage2-ingredient2-*"),
]

# Per-head Shannon attention entropy at the last answer token, in nats. The name
# changed between generation runs -- the intermediate pretraining checkpoints were
# produced before the span keys were suffixed with _end -- so both are accepted and
# the first one present in a checkpoint is used. They are the same measurement:
# the older files carry system/template/answer index keys where the newer ones
# carry system_end/template_end/answer_end.
#
# The newer runs also store renyi_answer_end (H_2) and trace_jacobian_answer_end
# (1 - exp(-H_2)); the latter is bounded to [0,1] and so cannot produce the nat-scale
# shifts the tables report. Only the newer checkpoints have them, so switching to
# either restricts the usable checkpoint list.
ENTROPY_KEYS = ("trace_jacobian_answer_end",)

# Number of result files read per checkpoint, not examples: each file holds ~5
# items, so 200 files ~ 1000 examples, which reproduces the group sizes printed in
# the paper (86 + 83 + 585 non-stayed-right at 1->2).
N_FILES = 200

# How an answer counts as correct.
CORRECT = lambda item: bool(item["exact_match"])
N_HEADS = 256


def _check_keys(stages=None, keys=ENTROPY_KEYS):
    """Fail before reading 2 GB of JSON if a checkpoint cannot supply the measure.

    The measures are not available everywhere. Only the newer generation runs write
    renyi_* and trace_jacobian_*; the intermediate pretraining checkpoints predate
    them and carry Shannon entropy alone, which cannot be converted -- tr(J) =
    1 - exp(-H_2) needs the Renyi-2 entropy, and H_1 does not determine H_2.
    """
    missing = []
    for name, pat in (stages or STAGES):
        d = _stage_dir(pat)
        f = min(glob.glob(f"{d}/results_20*.json"), key=lambda p: int(p.rsplit("_", 1)[-1][:-5]))
        with open(f) as fh:
            item = json.load(fh)[0]
        if _entropy_key(item, keys) is None:
            missing.append(f"{name} (has: {', '.join(k for k in item if k.startswith(('entropy_a', 'renyi_a', 'trace_jacobian_a')))})")
    if missing:
        raise KeyError(
            f"none of {keys} present in {len(missing)} of {len(stages or STAGES)} checkpoints:\n  "
            + "\n  ".join(missing))


def _stage_dir(pattern):
    hits = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not hits:
        raise FileNotFoundError(f"no checkpoint directory matching {pattern} under {ROOT}")
    return hits[-1]


def _entropy_key(item, keys=ENTROPY_KEYS):
    for k in keys:
        if item.get(k) is not None:
            return k
    return None


def _load_stage(pattern, n_files=N_FILES, entropy_keys=ENTROPY_KEYS):
    """{question_id: (entropy vector, exact_match)} for one checkpoint."""
    d = _stage_dir(pattern)
    files = [p for p in glob.glob(f"{d}/results_20*.json") if "vllm" not in p and "shap" not in p]
    files.sort(key=lambda p: int(p.rsplit("_", 1)[-1].removesuffix(".json")))
    files = files[:n_files]

    def read(fp):
        with open(fp) as fh:
            items = json.load(fh)
        rows = []
        for it in (items if isinstance(items, list) else [items]):
            k = _entropy_key(it, entropy_keys)
            if k is not None:
                rows.append((it["question_id"], np.asarray(it[k], dtype=float), bool(CORRECT(it))))
        return rows

    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(read, files):
            for qid, vec, ok in rows:
                out[qid] = (vec, ok)
    return out


def load(n_files=N_FILES, entropy_keys=ENTROPY_KEYS):
    """Entropies and correctness for the examples every checkpoint answered.

    Returns H with shape (stage, example, head) and Y with shape (stage, example).
    Examples are matched on question_id rather than on file order, so a checkpoint
    that dropped or reordered an item cannot silently shift the pairing.
    """
    _check_keys(STAGES, entropy_keys)
    per_stage = [_load_stage(pat, n_files, entropy_keys) for _, pat in STAGES]
    qids = sorted(set.intersection(*(set(s) for s in per_stage)))
    if not qids:
        raise RuntimeError("no question_id is present in all checkpoints")

    H = np.stack([np.stack([s[q][0] for q in qids]) for s in per_stage])
    Y = np.stack([np.array([s[q][1] for q in qids]) for s in per_stage])
    assert H.shape[2] == N_HEADS, f"expected {N_HEADS} heads, got {H.shape[2]}"
    # step0 writes non-finite entries for heads whose attention row is degenerate.
    # Every mean below is nan-aware, but a head that is non-finite in most examples
    # would still produce a meaningless column, so the count is reported rather than
    # silently absorbed.
    bad = ~np.isfinite(H)
    if bad.any():
        print(f"[warn] {bad.sum()} of {H.size} entropy values non-finite "
              f"({bad.any(axis=(1, 2)).sum()} checkpoint(s) affected); means skip them")
        H = np.where(bad, np.nan, H)
    print(f"{len(qids)} examples shared across {len(STAGES)} checkpoints "
          f"({', '.join(f'{n}: {Y[i].mean():.1%}' for i, (n, _) in enumerate(STAGES))} correct)")
    return H, Y, qids


# --- Table 1: entropy shift by how correctness changed (tab:flips) ------------

GROUPS = {
    "wrong->right": lambda a, b: ~a & b,
    "right->wrong": lambda a, b: a & ~b,
    "stayed wrong": lambda a, b: ~a & ~b,
}


def table_flips(H, Y, transitions=(1, 2, 3)):
    """Mean per-head entropy shift between consecutive checkpoints, by group.

    The shift is averaged over the examples in the group and over all 256 heads.
    `transitions` gives the earlier index of each pair; 0->1 is excluded by
    default, as in the paper.
    """
    rows = []
    for i in transitions:
        d = H[i + 1] - H[i]                       # (example, head)
        for name, sel in GROUPS.items():
            m = sel(Y[i], Y[i + 1])
            rows.append(dict(transition=f"{i}->{i+1}", group=name,
                             n=int(m.sum()), shift=float(np.nanmean(d[m])) if m.any() else np.nan))
    flips = pd.DataFrame(rows)

    # Pooled rows are the unweighted mean over transitions, so a transition with
    # more flips does not dominate; that is what the paper's Pooled block reports.
    pooled = flips.groupby("group")["shift"].mean()
    for name in GROUPS:
        flips.loc[len(flips)] = dict(transition="pooled", group=name, n=np.nan,
                                     shift=pooled[name])
    flips.loc[len(flips)] = dict(transition="pooled", group="difference", n=np.nan,
                                 shift=pooled["wrong->right"] - pooled["right->wrong"])

    # Per-head test behind the sentence in the text: pool each direction over the
    # transitions head by head, then pair the 256 heads. Signed-rank rather than a
    # t-test because the per-head shifts are not symmetric across heads.
    w2r = np.nanmean([np.nanmean((H[i + 1] - H[i])[(~Y[i]) & Y[i + 1]], axis=0) for i in transitions], axis=0)
    r2w = np.nanmean([np.nanmean((H[i + 1] - H[i])[Y[i] & (~Y[i + 1])], axis=0) for i in transitions], axis=0)
    ok = np.isfinite(w2r) & np.isfinite(r2w)
    stat, p = wilcoxon(w2r[ok], r2w[ok])
    wil = dict(heads_lower=int((w2r[ok] < r2w[ok]).sum()), n_heads=int(ok.sum()),
               W=float(stat), p=float(p))
    return flips, wil


# --- Table 2: entropy level against entropy change (tab:level-change) ---------

def table_level_change(H, Y, transitions=(0, 1, 2, 3)):
    """Per-head entropy level vs. change, on examples wrong at the earlier point.

    Correlating the change against the midpoint (H_i + H_{i+1}) / 2 rather than
    against H_i avoids the regression-to-the-mean artifact: measurement noise in
    H_i enters the change with the opposite sign and would manufacture a negative
    correlation on its own.
    """
    rows = []
    for i in transitions:
        m = ~Y[i]
        a, b = np.nanmean(H[i][m], axis=0), np.nanmean(H[i + 1][m], axis=0)   # (head,)
        dH, mid = b - a, (a + b) / 2
        ok = np.isfinite(dH) & np.isfinite(mid)
        rho, p = spearmanr(mid[ok], dH[ok])
        rows.append(dict(transition=f"{i}->{i+1}", n_wrong=int(m.sum()), n_heads=int(ok.sum()),
                         rho=rho, p=p, mean_dH=float(np.nanmean(dH))))
    return pd.DataFrame(rows)


# --- LaTeX -------------------------------------------------------------------

def latex_flips(flips, wil):
    tex = [r"\begin{table}", r"\centering", r"\small",
           r"\begin{tabular}{llrr}", r"\toprule",
           r"Transition & Group & $n$ & mean entropy \\", r" &  &  & shift \\", r"\midrule"]
    for t, blk in flips.groupby("transition", sort=False):
        head = "Pooled " if t == "pooled" else f"${t.replace('->', r'\to')}$"
        for j, (_, r) in enumerate(blk.iterrows()):
            label = head if j == 0 else " " * len(head)
            n = "" if pd.isna(r["n"]) else f"{int(r['n'])}"
            tex.append(f"{label} & {r['group'].replace('->', r'$\to$'):<15} & {n:<4}"
                       f"& ${r['shift']:+.4f}$ \\\\")
        tex.append(r"\midrule" if t != "pooled" else r"\bottomrule")
    tex += [r"\end{tabular}",
            r"\caption{Per-head change in attention entropy between consecutive checkpoints,",
            r"grouped by how each example's correctness changed. Checkpoints are the OLMo~2",
            r"releases: " + ", ".join(n for n, _ in STAGES) + r". Pooled over the three",
            f"transitions, wrong$\\to$right lies below right$\\to$wrong on {wil['heads_lower']} of",
            f"{wil['n_heads']} heads (Wilcoxon $W = {wil['W']:.0f}$, $p = {wil['p']:.1e}$)." + "}",
            r"\label{tab:flips}", r"\end{table}"]
    return "\n".join(tex)


def _sci(p):
    """p-value as LaTeX scientific notation, e.g. 4.5e-35 -> 5\\times10^{-35}."""
    if p == 0 or not np.isfinite(p):
        return r"<10^{-300}"
    mant, exp = f"{p:.0e}".split("e")
    return rf"{int(mant)}\times10^{{{int(exp)}}}"


def latex_level_change(lvl):
    tex = [r"\begin{table}", r"\centering", r"\small", r"\resizebox{\linewidth}{!}{%",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Transition & $\rho$ & $p$ & mean $\Delta H$ \\", r"\midrule"]
    for _, r in lvl.iterrows():
        a, b = r["transition"].split("->")
        tex.append(f"{a} $\\to$ {b} & ${r['rho']:+.3f}$ & ${_sci(r['p'])}$ & ${r['mean_dH']:+.3f}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"}",
            r"\caption{Per-head attention entropy: level against change, on examples the",
            r"model answers incorrectly at the earlier checkpoint. Spearman $\rho$ over",
            f"$m = {N_HEADS}$ heads, correlating $\\Delta H$ with the midpoint" + r" $(H_i + H_{i+1})/2$.}",
            r"\label{tab:level-change}", r"\end{table}"]
    return "\n".join(tex)


if __name__ == "__main__":
    H, Y, _ = load()
    flips, wil = table_flips(H, Y)
    lvl = table_level_change(H, Y)
    print("\n" + flips.to_string(index=False))
    print(f"\nwrong->right below right->wrong on {wil['heads_lower']}/{wil['n_heads']} heads "
          f"(W={wil['W']:.0f}, p={wil['p']:.2e})")
    print("\n" + lvl.to_string(index=False))
    print("\n" + "=" * 70 + "\n" + latex_flips(flips, wil))
    print("\n" + "=" * 70 + "\n" + latex_level_change(lvl))


# --- Teacher-forced variant: group by learning, not by correctness -------------
#
# Generation-based grouping is confounded on base checkpoints (see score_gold.py):
# at step480000 the model emits nothing for 54% of prompts, so a correctness flip
# largely records that it started producing text. These read the npz files written
# by score_gold.py, where both quantities come from one teacher-forced pass over
# prompt + gold answer, and group by the change in log P(gold) instead.

GOLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_scores")
GOLD_STAGES = [
    ("step480k",    "stage1-step480000-tokens1007B"),
    ("step1020k",   "stage1-step1020000-tokens2140B"),
    ("step1840k",   "stage1-step1840000-tokens3859B"),
    ("stage2-ing1", "stage2-ingredient1-step23852-tokens51B"),
    ("stage2-ing2", "stage2-ingredient2-step23852-tokens51B"),
]


def load_gold():
    """TJ[stage, example, head] and LP[stage, example] over the shared examples."""
    per = []
    for _, rev in GOLD_STAGES:
        z = np.load(os.path.join(GOLD_DIR, f"{rev}.npz"))
        per.append({q: (z["tj"][i], z["logp_mean"][i]) for i, q in enumerate(z["qids"])})
    qids = sorted(set.intersection(*(set(p) for p in per)))
    TJ = np.stack([np.stack([p[q][0] for q in qids]) for p in per])
    LP = np.stack([np.array([p[q][1] for q in qids]) for p in per])
    print(f"{len(qids)} examples x {len(GOLD_STAGES)} checkpoints; "
          f"log P(gold)/token " + " -> ".join(f"{l.mean():.3f}" for l in LP))
    return TJ, LP, qids


def table_learning(TJ, LP, transitions=(0, 1, 2, 3), q=0.1):
    """Per-head trace-Jacobian shift by how much the gold answer's likelihood moved.

    Replaces the correctness groups with deciles of the change in log P(gold):
    the top q are examples the model learned over the interval, the bottom q are
    ones it lost, and the middle is the control that replaces "stayed wrong".
    """
    rows = []
    for i in transitions:
        d_lp = LP[i + 1] - LP[i]
        d_tj = TJ[i + 1] - TJ[i]
        lo, hi = np.quantile(d_lp, q), np.quantile(d_lp, 1 - q)
        groups = {"learned": d_lp >= hi, "unlearned": d_lp <= lo,
                  "unchanged": (d_lp > lo) & (d_lp < hi)}
        for name, m in groups.items():
            rows.append(dict(transition=f"{i}->{i+1}", group=name, n=int(m.sum()),
                             shift=float(np.nanmean(d_tj[m]))))
    out = pd.DataFrame(rows)
    pooled = out.groupby("group")["shift"].mean()
    for name in ("learned", "unlearned", "unchanged"):
        out.loc[len(out)] = dict(transition="pooled", group=name, n=np.nan, shift=pooled[name])
    out.loc[len(out)] = dict(transition="pooled", group="difference", n=np.nan,
                             shift=pooled["learned"] - pooled["unlearned"])

    a = np.nanmean([np.nanmean((TJ[i+1]-TJ[i])[(LP[i+1]-LP[i]) >= np.quantile(LP[i+1]-LP[i], 1-q)], axis=0)
                    for i in transitions], axis=0)
    b = np.nanmean([np.nanmean((TJ[i+1]-TJ[i])[(LP[i+1]-LP[i]) <= np.quantile(LP[i+1]-LP[i], q)], axis=0)
                    for i in transitions], axis=0)
    ok = np.isfinite(a) & np.isfinite(b)
    stat, p = wilcoxon(a[ok], b[ok])
    wil = dict(heads_lower=int((a[ok] < b[ok]).sum()), n_heads=int(ok.sum()),
               W=float(stat), p=float(p))
    return out, wil


def table_level_change_gold(TJ, LP, transitions=(0, 1, 2, 3), frac=0.5):
    """Level against change, on the examples the model scores worst at step i.

    The correctness restriction becomes the bottom `frac` by log P(gold), which is
    the same idea -- examples the model has not learned yet -- without depending on
    whether it can emit them in the expected format.
    """
    rows = []
    for i in transitions:
        m = LP[i] <= np.quantile(LP[i], frac)
        a, b = np.nanmean(TJ[i][m], axis=0), np.nanmean(TJ[i + 1][m], axis=0)
        dH, mid = b - a, (a + b) / 2
        ok = np.isfinite(dH) & np.isfinite(mid)
        rho, p = spearmanr(mid[ok], dH[ok])
        rows.append(dict(transition=f"{i}->{i+1}", n=int(m.sum()), rho=rho, p=p,
                         mean_dTJ=float(np.nanmean(dH))))
    return pd.DataFrame(rows)


def table_learning_relative(H, LP, transitions=(0, 1, 2, 3), q=0.1):
    """Relative entropy change, in percent, by how much log P(gold) moved.

    Heads sit at very different entropy levels, so an absolute shift of 0.02 nats
    is a different event at H = 0.2 than at H = 2.5. Each head's change is divided
    by its own midpoint (H_i + H_{i+1}) / 2 -- the midpoint rather than H_i for the
    same reason table_level_change uses it: dividing by the earlier value lets
    measurement noise in H_i inflate the ratio and manufactures a trend.
    """
    rows = []
    for i in transitions:
        d_lp = LP[i + 1] - LP[i]
        rel = 100.0 * (H[i + 1] - H[i]) / np.clip((H[i] + H[i + 1]) / 2, 1e-9, None)
        lo, hi = np.quantile(d_lp, q), np.quantile(d_lp, 1 - q)
        for name, m in {"learned": d_lp >= hi, "unlearned": d_lp <= lo,
                        "unchanged": (d_lp > lo) & (d_lp < hi)}.items():
            rows.append(dict(transition=f"{i}->{i+1}", group=name, n=int(m.sum()),
                             rel_pct=float(np.nanmean(rel[m]))))
    out = pd.DataFrame(rows)
    pooled = out.groupby("group")["rel_pct"].mean()
    for name in ("learned", "unlearned", "unchanged"):
        out.loc[len(out)] = dict(transition="pooled", group=name, n=np.nan, rel_pct=pooled[name])
    out.loc[len(out)] = dict(transition="pooled", group="difference", n=np.nan,
                             rel_pct=pooled["learned"] - pooled["unlearned"])
    return out


def dose_response(H, LP, transitions=(0, 1, 2, 3)):
    """Continuous form: does entropy fall *more* the more an example is learned?

    Uses every example rather than the top and bottom deciles, so it tests the
    graded relationship directly and on ten times the data. One value per example
    (its relative entropy change averaged over the 256 heads) against its change in
    log P(gold). A negative rho is the prediction: more learning, larger fall.
    """
    rows = []
    for i in transitions:
        d_lp = LP[i + 1] - LP[i]
        rel = 100.0 * (H[i + 1] - H[i]) / np.clip((H[i] + H[i + 1]) / 2, 1e-9, None)
        per_ex = np.nanmean(rel, axis=1)
        ok = np.isfinite(per_ex) & np.isfinite(d_lp)
        rho, p = spearmanr(d_lp[ok], per_ex[ok])
        rows.append(dict(transition=f"{i}->{i+1}", n=int(ok.sum()), rho=rho, p=p,
                         mean_rel_pct=float(np.nanmean(per_ex))))
    return pd.DataFrame(rows)

