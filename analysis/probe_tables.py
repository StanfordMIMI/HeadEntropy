"""Appendix table: trained probes on the HeadEntropy vector, in-distribution AUROC.

Reads the latest latex_results_*.csv of every model in tables_<dataset>_trace_jacobian_answer/
and prints one LaTeX table: per dataset the AUROC on its validation split of a logistic
regression (he_logistic_regression_<layers>), an MLP (he_mlp) and XGBoost (he_xgboost), all
trained on the per-head HeadEntropy vector of the train split.

    python analysis/probe_tables.py      # the tabular; analysis/probe_table.tex wraps it with the caption
"""
import glob
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = [("triv", "TriviaQA"), ("hot", "HotpotQA"), ("indmed", "MedMCQA")]
MODELS = ["Qwen3-1.7B", "Llama-3.2-3B-Instruct", "Qwen3-8B", "Llama-3.1-8B-Instruct", "gemma-4-12b-it", "Qwen3-32B"]
PROBES = [("LR", re.compile(r"^he_logistic_regression_\d+$")), ("MLP", re.compile(r"^he_mlp$")), ("XGBoost", re.compile(r"^he_xgboost$"))]


def latest_csv(dataset, model):
    files = sorted(glob.glob(f"{ROOT}/tables_{dataset}_trace_jacobian_answer/{model}/latex_results_*.csv"))
    return files[-1] if files else None


def auroc(csv, pattern):
    df = pd.read_csv(csv)
    rows = df[df["name"].str.match(pattern)]
    return float(rows["roc_auc_test"].iloc[0]) if len(rows) else None


def table():
    rows = {}
    for model in MODELS:
        rows[model] = {}
        for key, name in DATASETS:
            csv = latest_csv(key, model)
            rows[model][name] = {p: (auroc(csv, pat) if csv else None) for p, pat in PROBES}
    return rows


def to_latex(rows):
    names = [n for _, n in DATASETS]
    out = ["\\begin{tabular}{l" + "ccc" * len(names) + "}", "\\toprule",
           "Model & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in names) + " \\\\",
           " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(names))),
           " & " + " & ".join(" & ".join(p for p, _ in PROBES) for _ in names) + " \\\\", "\\midrule"]
    for model, per_dataset in rows.items():
        cells = []
        for n in names:
            vals = per_dataset[n]
            best = max((v for v in vals.values() if v is not None), default=None)
            for p, _ in PROBES:
                v = vals[p]
                cells.append("--" if v is None else (f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"))
        out.append(model.replace("-Instruct", "").replace("-it", "") + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


if __name__ == "__main__":
    print(to_latex(table()))
