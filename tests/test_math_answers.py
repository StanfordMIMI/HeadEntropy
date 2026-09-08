import json
import os
import shutil
import tempfile
import unittest

from run.utils.math_evaluation import _normalize, math_equal
from run.utils.utils_build_prompts import extract_boxed_math
from run.utils.utils_generation import extract_answer, extract_math_answer


class TestExtractBoxedMath(unittest.TestCase):
    """The reference answer, read out of the dataset's worked solution."""

    def _boxed(self, solution):
        return extract_boxed_math({"solution": solution})

    def test_plain_and_nested(self):
        self.assertEqual(self._boxed(r"so it is $\boxed{42}$."), "42")
        self.assertEqual(self._boxed(r"$\boxed{\frac{1}{2}}$"), r"\frac{1}{2}")
        self.assertEqual(self._boxed(r"$\fbox{7}$"), "7")

    def test_last_box_wins(self):
        self.assertEqual(self._boxed(r"$\boxed{1}$ ... in fact $\boxed{2}$"), "2")

    def test_dollar_amount_keeps_its_digits(self):
        # \$20 used to lose the dollar and keep the backslash, leaving "\20"
        self.assertEqual(self._boxed(r"we win $\boxed{\$20}$"), "20")
        self.assertEqual(self._boxed(r"$\boxed{\$1.25}$"), "1.25")

    def test_presentation_macros_dropped(self):
        self.assertEqual(self._boxed(r"$\boxed{5\text{ cm}}$"), "5 cm")
        self.assertEqual(self._boxed(r"$\boxed{\dfrac{3}{4}}$"), r"\frac{3}{4}")

    def test_nothing_boxed(self):
        self.assertIsNone(self._boxed("the answer is 5"))


class TestExtractMathAnswer(unittest.TestCase):
    """The model's answer, read out of a generation."""

    def test_two_line_format(self):
        self.assertEqual(extract_math_answer("Answer: 42\nCertainty: 100"), "42")

    def test_negative_numbers_survive(self):
        # the general extractor reads these as a bare confidence and deletes them
        self.assertEqual(extract_answer("Answer: -2\nCertainty: 100"), "")
        self.assertEqual(extract_math_answer("Answer: -2\nCertainty: 100"), "-2")
        self.assertEqual(extract_math_answer("Answer: -2 Certainty: 100"), "-2")
        self.assertEqual(extract_math_answer("Answer: n-1\nCertainty: 90"), "n-1")

    def test_last_label_wins(self):
        text = ("<think>the format is Answer: [final answer only]. I get 3, no, 4."
                "</think>\n\nAnswer: 4\nCertainty: 60")
        self.assertEqual(extract_math_answer(text), "4")

    def test_delimiters_and_box_stripped(self):
        self.assertEqual(extract_math_answer(r"Answer: $\frac{1}{2}$"), r"\frac{1}{2}")
        self.assertEqual(extract_math_answer(r"Answer: $\boxed{30}$" + "\nCertainty: 100"), "30")
        self.assertEqual(extract_math_answer(r"Answer: \[x^2\]"), "x^2")

    def test_dollar_amount_is_not_a_delimiter(self):
        self.assertEqual(extract_math_answer(r"Answer: $\$20$"), r"\$20")

    def test_display_equation_on_the_next_lines(self):
        text = ("Answer: \n\\[\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}\\]\n"
                "Certainty: 100")
        self.assertEqual(extract_math_answer(text),
                         "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}")

    def test_unlabelled_generation(self):
        self.assertEqual(extract_math_answer("7\nCertainty: 100"), "7")

    def test_empty(self):
        self.assertEqual(extract_math_answer(""), "")
        self.assertEqual(extract_math_answer("Answer:\n"), "")


class TestMathEqual(unittest.TestCase):
    """Two ways of writing one answer have to compare equal."""

    def test_identical(self):
        self.assertTrue(math_equal("42", "42"))

    def test_fraction_forms(self):
        for pred in ("0.5", "1/2", r"\frac{1}{2}", r"\dfrac{1}{2}", r"\frac12"):
            with self.subTest(pred=pred):
                self.assertTrue(math_equal(pred, r"\frac{1}{2}"))
        self.assertTrue(math_equal(r"\frac9{19}", "9/19"))

    def test_reordering_and_radicals(self):
        self.assertTrue(math_equal(r"2\pi", r"\pi \cdot 2"))
        self.assertTrue(math_equal(r"\sqrt{4}", "2"))
        self.assertTrue(math_equal("2 \\sqrt{2}", "\\sqrt{8}"))

    def test_decorations_ignored(self):
        self.assertTrue(math_equal("90^\\circ", "90"))
        self.assertTrue(math_equal("118 dollars", "118"))
        self.assertTrue(math_equal("1,000", "1000"))
        self.assertTrue(math_equal("14,916", "14{,}916"))
        self.assertTrue(math_equal("x = 5", "5"))
        self.assertTrue(math_equal(r"\$20", "20"))

    def test_mixed_numbers(self):
        self.assertTrue(math_equal("3 1/3", r"\frac{10}{3}"))
        self.assertTrue(math_equal(r"1\frac{1}{10}", "1.1"))

    def test_a_variable_is_not_a_unit(self):
        # the unit stripper must not read the "m" of an algebraic answer as metres
        self.assertTrue(math_equal("m", "m"))
        self.assertFalse(math_equal("m", ""))

    def test_wrong_answers_stay_wrong(self):
        self.assertFalse(math_equal("-2", "2"))
        self.assertFalse(math_equal("1/4", r"\frac{3}{4}"))
        self.assertFalse(math_equal("", "5"))
        self.assertFalse(math_equal("5", ""))
        self.assertFalse(math_equal("[0,1]", "[0,1)"))

    def test_unparseable_falls_back_to_the_string(self):
        matrix = r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}"
        self.assertTrue(math_equal(matrix, matrix))
        self.assertFalse(math_equal(matrix, r"\begin{pmatrix} 2 \\ 1 \end{pmatrix}"))
        self.assertTrue(math_equal(r"(-\infty, 0) \cup (0, \infty)",
                                   r"(-\infty,0)\cup(0,\infty)"))


class _FakeCompletion:
    def __init__(self, text):
        self.text = text
        self.token_ids = [0, 1, 2]


class _FakeRequestOutput:
    def __init__(self, text):
        self.prompt_token_ids = [3, 4]
        self.outputs = [_FakeCompletion(text)]


class _FakeLLM:
    """Only what send_to_vllm touches: one canned generation per prompt."""

    def __init__(self, texts):
        self.texts = texts

    def generate(self, prompts, sampling_params):
        return [_FakeRequestOutput(t) for t in self.texts[:len(prompts)]]


class TestSendToVllmIsMath(unittest.TestCase):
    """--data_name hendrycks_math has to reach the answer extractor."""

    GENERATIONS = ["Answer: -2\nCertainty: 100",
                   "Answer: $\\boxed{\\frac{1}{2}}$\nCertainty: 80"]

    def _run(self, is_math):
        from datasets import Dataset
        from run.utils.utils_generation import send_to_vllm

        data = Dataset.from_dict({
            "messages": ["Question: a", "Question: b"],
            "ground_truth_answers": [["-2"], ["\\frac{1}{2}"]],
        })
        out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        paths = send_to_vllm(_FakeLLM(self.GENERATIONS), data, out_dir,
                             is_math=is_math)
        with open(paths[0]) as f:
            return json.load(f)

    def test_math_answers_survive_and_score(self):
        records = self._run(is_math=True)
        self.assertEqual([r["model_answer"] for r in records], ["-2", "\\frac{1}{2}"])
        for record in records:
            self.assertTrue(math_equal(record["model_answer"],
                                       record["ground_truth_answers"][0]))

    def test_general_extractor_is_untouched(self):
        records = self._run(is_math=False)
        self.assertEqual(records[0]["model_answer"], "")


if __name__ == "__main__":
    unittest.main()
