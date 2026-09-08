

import unittest

from run.utils.utils_generation import (
    _token_strings,
    compute_token_idx_llama,
    compute_token_idx_qwen,
)


class _FakeProcessor:
    """convert_ids_to_tokens, including the None a tokenizer returns for an id it
    has no entry for."""

    def __init__(self, name, vocab):
        self.name_or_path = name
        self.vocab = vocab

    def convert_ids_to_tokens(self, ids):
        return [self.vocab.get(i) for i in ids]


class TestTokenStrings(unittest.TestCase):
    def test_unnamed_id_becomes_empty(self):
        proc = _FakeProcessor("Qwen/Qwen3-1.7B", {0: "Ġhello", 1: None, 2: "Ċworld"})
        self.assertEqual(_token_strings([0, 1, 2], proc), ["hello", "", "world"])

    def test_strip_chars_are_configurable(self):
        proc = _FakeProcessor("google/gemma-4-12b-it", {0: "▁a", 1: None})
        self.assertEqual(_token_strings([0, 1], proc, strip_chars=("▁",)), ["a", ""])


class TestComputeTokenIdx(unittest.TestCase):
    """An id the tokenizer cannot name used to raise AttributeError on None."""

    QWEN = ["<|im_start|>", "system", "Please", "answer", "<|im_start|>", "user",
            "Question", "?", "<|im_start|>", "assistant", "<think>", "reasoning",
            "</think>", "Answer", "Ġ42", "Cert", "ainty", "Ġ100"]

    LLAMA = ["<|begin_of_text|>", "system", "<|end_header_id|>", "Please", "answer",
             "user", "<|end_header_id|>", "Question", "?", "assistant",
             "<|end_header_id|>", "Answer", "Ġ42", "Cert", "ainty", "Ġ100"]

    def _idx(self, fn, name, tokens, unnamed_at=None):
        vocab = dict(enumerate(tokens))
        if unnamed_at is not None:
            vocab[unnamed_at] = None
        return fn(list(range(len(tokens))), _FakeProcessor(name, vocab))

    def test_qwen_unnamed_id_does_not_raise(self):
        clean = self._idx(compute_token_idx_qwen, "Qwen/Qwen3-1.7B", self.QWEN)
        # the unnamed id sits inside the generated answer, as it did on FEVER
        holed = self._idx(compute_token_idx_qwen, "Qwen/Qwen3-1.7B", self.QWEN,
                          unnamed_at=11)
        self.assertEqual(clean["system_end"], 2)
        self.assertEqual(clean["answer_end"], 15)
        self.assertEqual(holed, clean)

    def test_llama_unnamed_id_does_not_raise(self):
        clean = self._idx(compute_token_idx_llama, "meta-llama/Llama-3.2-3B-Instruct",
                          self.LLAMA)
        holed = self._idx(compute_token_idx_llama, "meta-llama/Llama-3.2-3B-Instruct",
                          self.LLAMA, unnamed_at=12)
        self.assertEqual(clean["answer_end"], 13)
        self.assertEqual(holed, clean)


if __name__ == "__main__":
    unittest.main()
