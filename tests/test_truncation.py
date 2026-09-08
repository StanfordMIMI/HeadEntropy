

import unittest

from run.utils.utils_analyse import truncation_budget


def _run(lengths):
    return [{"output": {"usage": {"completion_tokens": n}}} for n in lengths]


class TestTruncationBudget(unittest.TestCase):
    def test_spike_at_the_budget_is_found(self):
        # 18% of Qwen3-1.7B's TriviaQA generations stop at exactly 768
        lengths = list(range(100, 200)) + [768] * 22
        self.assertEqual(truncation_budget(_run(lengths)), 768)

    def test_a_higher_budget_is_found_too(self):
        # the MATH rerun raised it to 1536; nothing about 768 is special
        self.assertEqual(truncation_budget(_run([50, 60, 70] + [1536] * 20)), 1536)

    def test_untruncated_run_has_no_budget(self):
        # Llama-3.2-3B answers TriviaQA in ~10 tokens: the longest one is not a cap
        lengths = [9] * 40 + [10] * 40 + [11] * 20 + [17]
        self.assertIsNone(truncation_budget(_run(lengths)))

    def test_a_lone_repeat_of_the_maximum_is_not_a_budget(self):
        # two items sharing the longest length out of 1000 is coincidence
        self.assertIsNone(truncation_budget(_run([5] * 998 + [40, 40])))

    def test_missing_and_malformed_usage(self):
        data = _run([768] * 20) + [{"output": {}}, {}, {"output": {"usage": {}}}]
        self.assertEqual(truncation_budget(data), 768)
        self.assertIsNone(truncation_budget([{}, {"output": {}}]))
        self.assertIsNone(truncation_budget([]))


if __name__ == "__main__":
    unittest.main()
