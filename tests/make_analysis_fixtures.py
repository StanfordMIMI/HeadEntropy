"""Build the input fixtures run_analyses.py is characterised against.

run_analyses.py wants seven result directories -- train, validation and five
generalization sets -- each holding files named results_20*_<n>.json. We only
have one real run (tests/fixtures/results_hf_qwen3-0.6b.json, produced by
tests/test_generation.py --regenerate), so the other six are simulated from it:
every float is jittered by ~2% and the labels are re-dealt per dataset, which
gives each set its own numbers and its own prevalence while keeping the schema,
the token-index landmarks and the feature dimensions exactly as the real
pipeline emits them.

Deterministic: a fixed seed per dataset, no wall-clock or hash randomness.

    python -m tests.make_analysis_fixtures
"""

import json
import os

import numpy as np

SOURCE = "tests/fixtures/results_hf_qwen3-0.6b.json"
# run_analyses derives its output directory as
# out_prefix_train.split("_")[1].split("/")[0], so the prefix has to keep the
# repo's "results_<dataset>/..." shape -- here that yields dataset suffix "tqa".
FIXTURE_ROOT = "tests/fixtures/analyses/results_tqa"
# name of the file inside each directory: get_data() globs results_20*.json and
# sorts on the integer after the last underscore.
RESULT_FILE = "results_20250625_225135_5.json"

# dataset -> (seed, label pattern). Prevalences differ so the AUC/ECE numbers
# are not accidentally identical across sets.
DATASETS = {
    "train": (0, [False, False, True, True, False]),
    "validation": (1, [False, True, False, True, False]),
    "med": (2, [True, False, False, True, False]),
    "hot": (3, [False, True, True, False, True]),
    "math": (4, [True, False, True, False, False]),
    "fever": (5, [False, False, False, True, True]),
    "temp": (6, [True, True, False, False, False]),
}

JITTER = 0.02  # relative sigma applied to every float leaf


def jitter_floats(value, rng):
    """Scale float leaves by ~N(1, JITTER); leave ints, bools and strings alone.

    Ints are the token-index landmarks (system_end <= template_end <= ... <=
    certainty_end) and the usage counts. Perturbing them would break the
    invariants get_entropy and compute_token_lengths rely on.
    """
    if isinstance(value, dict):
        return {k: jitter_floats(v, rng) for k, v in value.items()}
    if isinstance(value, list):
        return [jitter_floats(v, rng) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return value * float(rng.normal(1.0, JITTER))
    return value


def make_dataset(records, labels, rng):
    out = []
    for record, label in zip(records, labels):
        item = jitter_floats(record, rng)
        item["exact_match"] = bool(label)
        item["f1"] = 1.0 if label else 0.0
        # model_confidence is a numeric string consumed via safe_str_to_float
        item["model_confidence"] = str(int(rng.integers(50, 100)))
        out.append(item)
    return out


def main():
    with open(SOURCE, "r") as f:
        records = json.load(f)

    for name, (seed, labels) in DATASETS.items():
        assert len(labels) == len(records), f"{name}: need one label per record"
        assert 0 < sum(labels) < len(labels), f"{name}: needs both classes"
        rng = np.random.default_rng(seed)
        data = make_dataset(records, labels, rng)

        out_dir = os.path.join(FIXTURE_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, RESULT_FILE)
        with open(path, "w") as f:
            json.dump(data, f)
        size = os.path.getsize(path) / 1024
        print(f"  {name:11s} {len(data)} records, prevalence {sum(labels)/len(labels):.1f}, {size:7.0f} KB  {path}")


if __name__ == "__main__":
    main()
