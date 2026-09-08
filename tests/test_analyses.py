
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "run" / "run_analyses.py"
FIXTURES = "tests/fixtures/analyses/results_tqa"
GOLDEN = REPO / "tests" / "fixtures" / "analyses_golden.json"

MODEL_NAME = "qwen3-0.6b"
SEMANTIC_SECTION = "trace_jacobian_answer_end"
# where run_analyses puts things: tables_<suffix>_<section>/<model_name>
OUT_DIR = f"tables_tqa_{SEMANTIC_SECTION.replace('_end', '')}/{MODEL_NAME}"

CLI_ARGS = [
    "--out_prefix_train", f"{FIXTURES}/train",
    "--out_prefix_val", f"{FIXTURES}/validation",
    "--model_name", MODEL_NAME,
    "--med_generalization", f"{FIXTURES}/med",
    "--hot_generalization", f"{FIXTURES}/hot",
    "--math_generalization", f"{FIXTURES}/math",
    "--fever_generalization", f"{FIXTURES}/fever",
    "--temp_generalization", f"{FIXTURES}/temp",
    "--compute_ci", "True",
    "--semantic_section", SEMANTIC_SECTION,
]

REL_TOL = float(os.environ.get("HE_TEST_REL_TOL", "1e-3"))
ABS_TOL = float(os.environ.get("HE_TEST_ABS_TOL", "1e-6"))

TIMESTAMP_RE = re.compile(r"_20\d{6}_\d{6}")


def _strip_timestamp(name):
    """latex_results_20260804_213637.csv -> latex_results_<ts>.csv"""
    return TIMESTAMP_RE.sub("_<ts>", name)


def _prepare_script(work_dir):
    src = SCRIPT.read_text()
    patched = re.sub(r"\n(\s*)sys\.exit\(0\)\n",
                     r"\n\1# sys.exit(0)  # neutralised by tests/test_analyses.py\n",
                     src, count=1)
    target = work_dir / "run_analyses_under_test.py"
    target.write_text(patched)
    return target, patched != src


def run_analyses(work_dir):
    work_dir = Path(work_dir)
    os.symlink(REPO / "tests", work_dir / "tests")

    script, patched = _prepare_script(work_dir)
    env = dict(os.environ, PYTHONPATH=str(REPO), MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(script), *CLI_ARGS],
        cwd=work_dir, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"run_analyses failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")

    out_root = work_dir / OUT_DIR
    if not out_root.is_dir():
        raise RuntimeError(f"expected output dir {OUT_DIR}, got: "
                           f"{[p.name for p in work_dir.iterdir()]}")

    outputs = {}
    for path in sorted(out_root.rglob("*")):
        if not path.is_file():
            continue
        key = _strip_timestamp(path.name)
        if path.suffix == ".json":
            outputs[key] = json.loads(path.read_text())
        elif path.suffix == ".csv":
            rows = [r for r in path.read_text().splitlines() if r.strip()]
            header = rows[0].split(",")
            outputs[key] = [dict(zip(header, r.split(","))) for r in rows[1:]]
        elif path.suffix == ".txt":
            outputs[key] = path.read_text()
        else:
            outputs[key] = None  # figures: presence only
    return outputs, patched


def regenerate_golden():
    with tempfile.TemporaryDirectory() as work_dir:
        outputs, patched = run_analyses(work_dir)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(outputs, indent=1, sort_keys=True))
    print(f"wrote {GOLDEN.relative_to(REPO)}: {len(outputs)} output files, "
          f"{GOLDEN.stat().st_size / 1024:.0f} KB")
    if patched:
        print("note: sys.exit(0) was commented out in the copy under test")


class TestRunAnalyses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (REPO / FIXTURES / "train").is_dir():
            raise unittest.SkipTest(
                f"missing {FIXTURES}; build it with: python -m tests.make_analysis_fixtures")
        if not GOLDEN.exists():
            raise unittest.SkipTest(
                f"missing {GOLDEN.name}; record it with: python -m tests.test_analyses --regenerate")
        cls.golden = json.loads(GOLDEN.read_text())
        cls.work_dir = tempfile.mkdtemp(prefix="he_analyses_")
        cls.outputs, _ = run_analyses(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _assert_close(self, got, want, path):
        if isinstance(want, dict):
            self.assertIsInstance(got, dict, path)
            self.assertEqual(set(got), set(want), f"key sets differ at {path}")
            for key in want:
                self._assert_close(got[key], want[key], f"{path}.{key}")
        elif isinstance(want, list):
            self.assertIsInstance(got, list, path)
            self.assertEqual(len(got), len(want), f"length differs at {path}")
            for i, (g, w) in enumerate(zip(got, want)):
                self._assert_close(g, w, f"{path}[{i}]")
        elif isinstance(want, str):
            # CSV cells arrive as strings; compare numerically when they parse
            try:
                gf, wf = float(got), float(want)
            except (TypeError, ValueError):
                self.assertEqual(got, want, f"{path} differs")
                return
            self._assert_number(gf, wf, path)
        elif isinstance(want, bool) or want is None:
            self.assertEqual(got, want, f"{path} differs")
        elif isinstance(want, (int, float)):
            self._assert_number(got, want, path)
        else:
            self.assertEqual(got, want, f"{path} differs")

    def _assert_number(self, got, want, path):
        if got != got and want != want:  # both NaN: the script emits these
            return
        self.assertFalse(got != got, f"{path} became NaN (was {want!r})")
        self.assertFalse(want != want, f"{path} was NaN, is now {got!r}")
        self.assertTrue(
            abs(got - want) <= max(ABS_TOL, REL_TOL * abs(want)),
            f"{path} differs: {got!r} != {want!r} (rel_tol={REL_TOL}, abs_tol={ABS_TOL})",
        )

    def test_same_files_produced(self):
        self.assertEqual(sorted(self.outputs), sorted(self.golden))

    def test_results_table_unchanged(self):
        key = next(k for k in self.golden if k.startswith("latex_results"))
        got, want = self.outputs[key], self.golden[key]
        self.assertEqual([r["name"] for r in got], [r["name"] for r in want],
                         "the set or order of analysis rows changed")
        for got_row, want_row in zip(got, want):
            with self.subTest(row=want_row["name"]):
                self._assert_close(got_row, want_row, want_row["name"])

    def test_json_dumps_unchanged(self):
        for key, want in sorted(self.golden.items()):
            if not key.endswith(".json"):
                continue
            with self.subTest(output=key):
                self._assert_close(self.outputs[key], want, key)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                    help="record the current outputs as the golden baseline")
    known, rest = ap.parse_known_args()
    if known.regenerate:
        regenerate_golden()
    else:
        unittest.main(argv=[sys.argv[0], *rest])
