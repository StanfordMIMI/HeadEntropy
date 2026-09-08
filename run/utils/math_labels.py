"""Correct MATH reference answers at analysis time.

The MATH runs on disk were generated with an ``extract_boxed_math`` that matched only
``\\boxed{<digits>}`` and returned the string ``"No answer"`` for everything else. Since
that sentinel is not ``None``, run_generation's "drop the items with no reference"
filter kept them, so 42% of the items carry a reference that can never be matched and
the remaining 58% are exactly the problems whose answer is a non-negative integer --
a subset skewed towards Number Theory and away from Precalculus.

The reference answer is a property of the dataset, not of the generation, so it can be
put right here without generating anything again: this module rebuilds the split in the
order run_generation.py iterates it and reads each solution with the current extractor.

Records are looked up by their position in that split, which is what send_to_vllm
stores as ``question_id`` for a dataset without an id column. The problem text held by
the record is only its first line, so it cannot serve as the key -- but it is checked
against the problem found at that position, and a mismatch is reported rather than
silently scored.
"""

import re

from run.utils.utils_build_prompts import extract_boxed_math

# the seven configs run_generation.py concatenates for this dataset, in its order
_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
_DATASET = "EleutherAI/hendrycks_math"
_SEED = 42            # run_generation.py: data.shuffle(seed=42)
_NUM_EXAMPLES = 30000  # run_generation.py: --num_examples default

_by_index = None      # position -> (first line of the problem, [answers])
_by_first_line = None  # first line -> [answers], only where it is unambiguous


def _key(text):
    """The first line of a problem, stripped of the chat template and of whitespace."""
    text = re.split(r"<\|", str(text), maxsplit=1)[0]
    return re.sub(r"\s+", " ", text.split("\n")[0]).strip()


def _build():
    from datasets import concatenate_datasets, load_dataset

    try:
        data = concatenate_datasets(
            [load_dataset(_DATASET, split="test", name=c) for c in _CONFIGS]
        )
    except Exception as exc:                                  # offline, cache cleared
        raise SystemExit(
            f"MATH relabelling needs {_DATASET}, which could not be loaded ({exc}).\n"
            "The references stored in the results are unusable for 42% of the items, so "
            "the analysis stops rather than report numbers computed against them."
        )
    data = data.shuffle(seed=_SEED)
    data = data.select(range(min(_NUM_EXAMPLES, len(data))))

    by_index, first_lines = [], {}
    for item in data:
        answer = extract_boxed_math(item)
        line = _key(item["problem"])
        by_index.append((line, [answer] if answer is not None else None))
        if answer is not None:
            first_lines.setdefault(line, set()).add(answer)
    # a first line shared by two problems with different answers cannot identify either
    by_first_line = {k: sorted(v) for k, v in first_lines.items() if len(v) == 1}
    print(f"MATH labels: {len(by_index)} problems read from {_DATASET} "
          f"({len(by_first_line)} identifiable by their first line)")
    return by_index, by_first_line


def gold_answers(question, question_id=None):
    """The boxed answer(s) of this MATH problem, or None when it cannot be identified."""
    global _by_index, _by_first_line
    if _by_index is None:
        _by_index, _by_first_line = _build()

    line = _key(question)
    if isinstance(question_id, int) and 0 <= question_id < len(_by_index):
        expected, answers = _by_index[question_id]
        # the record holds only the first line, so the match is a prefix test
        if answers is not None and (expected == line or expected.startswith(line)):
            return answers
    return _by_first_line.get(line)
