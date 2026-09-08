
"""
Run with:
    python -m unittest tests.test_generation
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import glob
import json
import math
import re
import shutil
import tempfile
import unittest
from tqdm import tqdm
from run.utils.utils_generation import send_to_vllm, send_to_hf, process_single_output, compute_evals

# Checkpoint under test. Everything model-dependent (feature values, generated
# text) is compared against fixtures regenerated for this exact checkpoint.
MODEL_NAME = os.environ.get("HE_TEST_MODEL", "Qwen/Qwen3-0.6B")
# Checkpoint tests/results_vllm_0.json itself was generated with.
PIPELINE_MODEL = "Qwen/Qwen3-8B"

FIXTURE_DIR = "tests/fixtures"
_SLUG = MODEL_NAME.split("/")[-1].lower()
VLLM_FIXTURE = f"{FIXTURE_DIR}/results_vllm_{_SLUG}.json"
HF_FIXTURE = f"{FIXTURE_DIR}/results_hf_{_SLUG}.json"
REGENERATE_HINT = f"regenerate with: HE_TEST_MODEL={MODEL_NAME} python -m tests.test_decoding --regenerate"

# send_to_hf only flushes a file every 5 records, so keep this a multiple of 5.
N_HF_ITEMS = int(os.environ.get("HE_TEST_N_HF_ITEMS", "5"))
# Re-generating is the slow part, so only a couple of items go through send_to_vllm.
N_VLLM_ITEMS = int(os.environ.get("HE_TEST_N_VLLM_ITEMS", "2"))
TIMESTEMP = "20250625_225135"
# bf16/fp16 attention and hidden states are not reproducible to the last bit
# across GPUs or transformers versions, so floats are compared with a tolerance.
REL_TOL = float(os.environ.get("HE_TEST_REL_TOL", "1e-3"))
ABS_TOL = float(os.environ.get("HE_TEST_ABS_TOL", "1e-4"))


class _FakeCompletionOutput:
    """Mimics vllm.CompletionOutput."""

    def __init__(self, text, token_ids):
        self.text = text
        self.token_ids = token_ids


class _FakeRequestOutput:
    """Mimics vllm.RequestOutput (only the fields send_to_vllm touches)."""

    def __init__(self, prompt_token_ids, text, token_ids):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [_FakeCompletionOutput(text, token_ids)]


class HFGreedyLLM:
    """Stand-in for vllm.LLM backed by plain transformers .generate().

    Lets the tests exercise the real send_to_vllm plumbing (chunking, field
    extraction, file writing) without spinning up a second engine that would
    have to hold the weights again.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def generate(self, prompts, sampling_params):
        pad_id = self.processor.pad_token_id
        if pad_id is None:
            pad_id = self.processor.eos_token_id
        outputs = []
        for text in tqdm(prompts, desc="HFGreedyLLM.generate"):
            enc = self.processor(text, return_tensors="pt").to(self.model.device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                sequences = self.model.generate(
                    **enc,
                    max_new_tokens=sampling_params.max_tokens,
                    do_sample=sampling_params.temperature > 0,
                    temperature=sampling_params.temperature or None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=pad_id,
                )
            answer_ids = sequences[0, prompt_len:].tolist()
            outputs.append(
                _FakeRequestOutput(
                    prompt_token_ids=enc["input_ids"][0].tolist(),
                    # skip_special_tokens=True round-trips the stored "output" exactly
                    text=self.processor.decode(answer_ids, skip_special_tokens=True),
                    token_ids=answer_ids,
                )
            )
        return outputs


def load_model_and_processor(model_name=MODEL_NAME):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map="auto",  # {"": 0} if torch.cuda.is_available() else None, # "auto", #
        attn_implementation="eager",
        # cache_dir=f"{root}/hf_model_tmp_new",
        revision=None,
        trust_remote_code=True,
    )
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_name)


def load_generated_data():
    """The real vLLM run that feeds send_to_hf."""
    with open("./tests/results_vllm_0.json", "r") as f:
        return json.load(f)


def run_send_to_vllm(model, processor, source, out_dir):
    """send_to_vllm over `source`, returning the records it wrote."""
    from datasets import Dataset

    os.makedirs(out_dir, exist_ok=True)  # send_to_vllm assumes the dir exists
    data = Dataset.from_list([
        {"id": item["question_id"],
         "messages": item["prompt"],
         "ground_truth_answers": item["ground_truth_answers"]}
        for item in source
    ])
    paths = send_to_vllm(
        HFGreedyLLM(model, processor),
        data,
        out_dir,
        batch_size=len(data),
        temperature=0,
    )
    assert paths == [f"{out_dir}/results_vllm_0.json"], paths
    with open(paths[0], "r") as f:
        return json.load(f)


def run_send_to_hf(model, processor, source, out_dir):
    """send_to_hf over `source`, returning the records it flushed to disk."""
    os.makedirs(out_dir, exist_ok=True)  # send_to_hf assumes the dir exists
    send_to_hf(
        model=model,
        processor=processor,
        generated_data=source,
        timestemp=TIMESTEMP,
        all_token_ids=False,
        out_dir=out_dir,
        start_idx=0,
    )
    # send_to_hf flushes every 5th record to results_{timestemp}_{count}.json
    records = []
    for path in sorted(glob.glob(f"{out_dir}/results_{TIMESTEMP}_*.json"),
                       key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0])):
        with open(path, "r") as f:
            records.extend(json.load(f))
    return records


def regenerate_fixtures():
    """Rebuild both expected-output fixtures with MODEL_NAME."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    model, processor = load_model_and_processor()
    generated_data = load_generated_data()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[regenerate] send_to_vllm over {N_VLLM_ITEMS} items with {MODEL_NAME}")
        vllm_records = run_send_to_vllm(
            model, processor, generated_data[:N_VLLM_ITEMS], os.path.join(tmp_dir, "vllm")
        )
        with open(VLLM_FIXTURE, "w") as f:
            json.dump(vllm_records, f, indent=2)
        print(f"[regenerate] wrote {VLLM_FIXTURE} ({len(vllm_records)} records)")

        print(f"[regenerate] send_to_hf over {N_HF_ITEMS} items with {MODEL_NAME}")
        hf_records = run_send_to_hf(
            model, processor, generated_data[:N_HF_ITEMS], os.path.join(tmp_dir, "hf")
        )
        with open(HF_FIXTURE, "w") as f:
            json.dump(hf_records, f)
        print(f"[regenerate] wrote {HF_FIXTURE} ({len(hf_records)} records)")


@unittest.skipUnless(torch.cuda.is_available(), "feature extraction runs on GPU")
class TestAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.processor = load_model_and_processor()
        cls.generated_data = load_generated_data()
        cls.timestemp = TIMESTEMP
        cls.all_token_ids = False
        cls.out_dir = "tests/tmp_results"
        cls.start_idx = 0
        # compare out of each function with the fixtures regenerated for this model
        cls.results = cls._load_fixture(HF_FIXTURE)
        cls.vllm_results = cls._load_fixture(VLLM_FIXTURE)

        # feature widths are derived from the config so any checkpoint works
        cls.num_layers = cls.model.config.num_hidden_layers
        cls.num_heads = cls.model.config.num_attention_heads
        cls.hidden_size = cls.model.config.hidden_size

        os.makedirs(cls.out_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.cuda.empty_cache()
        # only ever remove the scratch dir this test created
        if os.path.abspath(cls.out_dir).startswith(os.path.abspath("tests")):
            shutil.rmtree(cls.out_dir, ignore_errors=True)

    @staticmethod
    def _load_fixture(path):
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------ helpers

    def _hf_forward(self, item):
        """The forward pass send_to_hf runs, packed into its hf_out dict.
        """
        seq_tensor = torch.tensor(
            item["prompt_token_ids"] + item["answer_token_ids"], dtype=torch.long
        )
        pad_id = self.processor.pad_token_id
        if pad_id is None:
            pad_id = self.processor.eos_token_id
        full_token_ids = seq_tensor.unsqueeze(0).to(self.model.device)
        attention_mask = (full_token_ids != pad_id).long().to(self.model.device)
        with torch.no_grad():
            gen_out = self.model(
                input_ids=full_token_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_scores=True,
                output_hidden_states=True,
            )
        device = torch.device("cuda")
        attentions = torch.stack(gen_out["attentions"], dim=0)  # [L, B, H, T, T]
        hidden_s = gen_out["hidden_states"]
        hidden_states = torch.empty(
            (len(hidden_s), *hidden_s[0].shape), dtype=torch.float16
        )  # [L+1, B, T, D] CPU tensor
        for idx_h, h in enumerate(hidden_s):
            hidden_states[idx_h].copy_(h.to(dtype=torch.float16))
        return {
            "logits": gen_out["logits"][0],
            "attentions": attentions[:, -1].to(device),
            "attention_mask": attention_mask[0],
            "hidden_states": hidden_states[-1, -1].to(device),
            "hidden_states_all": hidden_states[:, -1].to(device),
        }

    def _assert_close(self, got, want, path="record"):
        """Deep comparison: exact for keys/strings/ints/bools, tolerant for floats."""
        if isinstance(want, dict):
            self.assertIsInstance(got, dict, path)
            self.assertEqual(set(got), set(want), f"key sets differ at {path}. {REGENERATE_HINT}")
            for key in want:
                self._assert_close(got[key], want[key], f"{path}.{key}")
        elif isinstance(want, list):
            self.assertIsInstance(got, list, path)
            self.assertEqual(len(got), len(want), f"length differs at {path}")
            for i, (g, w) in enumerate(zip(got, want)):
                self._assert_close(g, w, f"{path}[{i}]")
        elif isinstance(want, bool) or want is None or isinstance(want, str):
            # token counts, landmarks, exact_match, decoded text: no slack
            self.assertEqual(got, want, f"{path} differs")
        elif isinstance(want, int) and isinstance(got, int):
            self.assertEqual(got, want, f"{path} differs")
        elif isinstance(want, (int, float)):
            self.assertIsInstance(got, (int, float), f"{path} is not numeric")
            # isclose is False for NaN on either side, which is what we want here
            self.assertTrue(
                math.isclose(got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                f"{path} differs: {got!r} != {want!r} "
                f"(rel_tol={REL_TOL}, abs_tol={ABS_TOL})",
            )
        else:
            self.assertEqual(got, want, f"{path} differs")

    def _assert_finite(self, name, values):
        present = [v for v in values if v is not None]
        self.assertTrue(present, f"{name} is empty / all None")
        bad = [v for v in present if v != v or v in (float("inf"), float("-inf"))]
        self.assertFalse(bad, f"{name} contains non-finite entries: {bad[:5]}")

    def _assert_record_matches_source(self, record, source, idx):
        """Fixture-independent invariants of a feature record vs. its vllm input."""
        len_prompt = len(source["prompt_token_ids"])
        len_answer = len(source["answer_token_ids"])
        seq_len = len_prompt + len_answer

        # --- pass-through fields survive the round trip untouched
        for key in ("question_id", "prompt", "question", "model_answer",
                    "model_confidence", "ground_truth_answers"):
            self.assertEqual(record[key], source[key], f"{key} changed for item {idx}")
        # the large id arrays are stripped from the record
        self.assertNotIn("prompt_token_ids", record)
        self.assertNotIn("answer_token_ids", record)

        # --- output block
        out = record["output"]
        self.assertEqual(out["id"], idx, "output.id is the running index send_to_hf passes in")
        self.assertEqual(out["message"], source["output"])
        self.assertEqual(
            out["usage"],
            {"completion_tokens": len_answer,
             "prompt_tokens": len_prompt,
             "total_tokens": seq_len},
        )
        # compute_logprobs is fed the full [T, V] logits, so it spans prompt+answer
        self.assertEqual(len(out["logprobs"]), seq_len)
        for entry in out["logprobs"]:
            self.assertLessEqual(entry["logprob"], 1e-4)
            self.assertGreaterEqual(entry["entropy"], 0.0)

        # --- token index landmarks, in the order compute_token_idx_qwen asserts
        self.assertLessEqual(record["system_end"], record["template_end"])
        self.assertLessEqual(record["template_end"], record["question_end"])
        self.assertLessEqual(record["question_end"], record["answer_end"])
        self.assertLessEqual(record["answer_end"], record["certainty_end"])
        self.assertEqual(record["certainty_end"], seq_len)
        self.assertLess(record["question_end"], seq_len,
                        "lookback lens needs at least one token past the prompt")
        self.assertTrue([k for k in record if k.startswith("think_chunk_")],
                        "expected at least one think chunk landmark")

        # --- feature vectors, sized off the model config
        lh = self.num_layers * self.num_heads
        self.assertEqual(len(record["lookback_lens_ratio"]), lh)
        self._assert_finite("lookback_lens_ratio", record["lookback_lens_ratio"])
        self.assertEqual(len(record["attn_eig_prod"]), self.num_layers)
        self._assert_finite("attn_eig_prod", record["attn_eig_prod"])
        self.assertEqual(len(record["icr_score"]), self.num_layers)
        self.assertEqual(len(record["trace_jacobian_all"]), lh)
        self._assert_finite("trace_jacobian_all", record["trace_jacobian_all"])
        for key in ("system_end", "template_end", "question_end", "answer_end", "certainty_end"):
            self.assertEqual(len(record[f"embedding_{key}"]), self.hidden_size)
            self.assertEqual(len(record[f"trace_jacobian_{key}"]), lh)

        # --- evals are recomputed from the answer, not copied
        self.assertEqual(
            {"exact_match": record["exact_match"], "f1": record["f1"],
             "ground_truth_answers": record["ground_truth_answers"]},
            compute_evals(source["model_answer"], source["ground_truth_answers"]),
        )

    # ------------------------------------------------------------------- tests

    def test_send_to_vllm(self):
        # test whether the outputs match  self.generated_data
        # try to test without vllm just normal huggingface: HFGreedyLLM replays the
        # interface send_to_vllm expects, so the function under test is the real one.
        self.assertIsNotNone(self.vllm_results, f"missing {VLLM_FIXTURE}. {REGENERATE_HINT}")

        source = self.generated_data[:N_VLLM_ITEMS]
        out_dir = os.path.join(self.out_dir, "vllm")
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        written = run_send_to_vllm(self.model, self.processor, source, out_dir)

        self.assertEqual(len(written), len(source))
        for idx, (got, want, expected) in enumerate(zip(written, source, self.vllm_results)):
            with self.subTest(item=idx):
                # everything that does not depend on the decoder must match the
                # real vLLM run in tests/results_vllm_0.json exactly
                self.assertEqual(set(got), set(want), "key set differs")
                for key in ("question_id", "prompt", "question", "ground_truth_answers",
                            "prompt_token_ids"):
                    self.assertEqual(got[key], want[key], f"{key} differs")

                # the generation itself is a regression check against this model's fixture
                self._assert_close(got, expected, path=f"written[{idx}]")

                # ... and is reported against the real vLLM run, which HF greedy
                # search only tracks for a while (different kernels / batching)
                got_ids, want_ids = list(got["answer_token_ids"]), list(want["answer_token_ids"])
                diverged = next(
                    (i for i, (a, b) in enumerate(zip(got_ids, want_ids)) if a != b),
                    min(len(got_ids), len(want_ids)),
                )
                print(f"[item {idx}] HF({MODEL_NAME}) and vLLM({PIPELINE_MODEL}) token ids "
                      f"agree for {diverged} tokens (HF {len(got_ids)}, vLLM {len(want_ids)})")

                # the answer/confidence fields are pure string surgery on "output"
                self.assertEqual(got["output"],
                                 self.processor.decode(got_ids, skip_special_tokens=True))
                # split() hands back the whole generation when the label is absent, so
                # these hold only where the model actually emitted one
                if "Answer: " in got["output"]:
                    self.assertEqual(
                        got["model_answer"],
                        got["output"].split("Answer: ")[-1].split("Certainty: ")[0].strip())
                if "Certainty: " in got["output"]:
                    self.assertEqual(got["model_confidence"],
                                     got["output"].split("Certainty: ")[-1].strip())

    def test_send_to_hf(self):
        # test whether the outputs match  self.results
        # at the end delete tmp folder saved to self.out_dir
        self.assertIsNotNone(self.results, f"missing {HF_FIXTURE}. {REGENERATE_HINT}")

        out_dir = os.path.join(self.out_dir, "hf")
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        source = self.generated_data[:N_HF_ITEMS]
        records = run_send_to_hf(self.model, self.processor, source, out_dir)

        self.assertTrue(
            records,
            f"send_to_hf wrote nothing: it only flushes every 5 records and got {len(source)}",
        )
        self.assertEqual(len(records), 5 * (len(source) // 5))
        self.assertEqual(len(records), len(self.results), REGENERATE_HINT)

        for idx, (record, expected) in enumerate(zip(records, self.results)):
            with self.subTest(item=idx):
                self._assert_record_matches_source(record, source[idx], idx)
                self._assert_close(record, expected, path=f"records[{idx}]")

    def test_process_single_output(self):
        # basically a subroutine of test_send_to_hf
        self.assertIsNotNone(self.results, f"missing {HF_FIXTURE}. {REGENERATE_HINT}")

        source = self.generated_data[0]
        hf_out = self._hf_forward(source)

        record = process_single_output(
            hf_out=hf_out,
            vllm_out=source,
            processor=self.processor,
            all_token_ids=self.all_token_ids,
            idx=0,
        )
        self.assertIsNotNone(record)
        self._assert_record_matches_source(record, source, 0)
        # the same record send_to_hf writes for the first item
        self._assert_close(record, self.results[0])

        # idx=None falls back to the id carried by the vllm record
        record_no_idx = process_single_output(
            hf_out=hf_out, vllm_out=source, processor=self.processor,
            all_token_ids=self.all_token_ids, idx=None,
        )
        self.assertEqual(record_no_idx["output"]["id"], source["question_id"])

        # a logits/token-id length mismatch is skipped, not raised
        short = dict(source)
        short["answer_token_ids"] = source["answer_token_ids"][:-1]
        self.assertIsNone(
            process_single_output(hf_out=hf_out, vllm_out=short, processor=self.processor,
                                  all_token_ids=self.all_token_ids, idx=0)
        )

        # non-square attentions are a hard error
        bad = dict(hf_out)
        bad["attentions"] = hf_out["attentions"][..., :-1]
        with self.assertRaises(ValueError):
            process_single_output(hf_out=bad, vllm_out=source, processor=self.processor,
                                  all_token_ids=self.all_token_ids, idx=0)

        # an empty ground truth must not silently produce an eval
        no_gt = dict(source)
        no_gt["ground_truth_answers"] = []
        with self.assertRaises(ValueError):
            process_single_output(hf_out=hf_out, vllm_out=no_gt, processor=self.processor,
                                  all_token_ids=self.all_token_ids, idx=0)


# saved for later in different file
    # def test_compute_hidden_states(self):
    # def test_compute_lookback_lens(self):
    # def test_compute_trace_jacobian(self):
    # def test_compute_icr(self):
    # def test_compute_compute_attn_eig_prod(self)


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        regenerate_fixtures()
    else:
        unittest.main()
