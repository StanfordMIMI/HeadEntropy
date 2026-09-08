"""Report which generation runs behind run_*.sh are still missing or incomplete.

Reads the ROWS of each run script, resolves each run's directory the way env.sh's
run_dir does (the directory with the most feature files among those matching
<results_dir>/*temp<T>_<model>_<split><suffix>), and counts the records in it.

    python check_runs.py               # every run of every script
    python check_runs.py --todo        # only the unfinished ones
    python check_runs.py run_tqa.sh    # one script
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = ["run_tqa.sh", "run_hqa.sh", "run_indqa.sh", "run_mathqa.sh", "run_feverqa.sh"]

# examples the HF stage has to produce per run
TRAIN_TARGET = 10000
# size of the split run_generation.py actually iterates over (after its own filtering)
SPLIT_TARGET = {
    ("mandarjoshi/trivia_qa", "validation"): 17944,
    ("hotpotqa/hotpot_qa", "validation"): 7405,
    ("openlifescienceai/medmcqa", "validation"): 4183,
    ("pietrolesci/nli_fever", "validation"): 19998,
    ("EleutherAI/hendrycks_math", "test"): 5000,
}


def parse_script(path):
    """Settings and (model, split) rows of one run script."""
    text = open(path).read()
    env = {}
    for m in re.finditer(r'^(?:export\s+)?([A-Z_]+)=("?)([^"\n#]*)\2', text, re.M):
        env[m.group(1)] = m.group(3).strip()
    # the model argument nests quotes ("$(basename "$MODEL_NAME")"), hence the lazy .*?
    m = re.search(r'run_dir\s+"\$RESULTS_DIR"\s+"\$TEMPERATURE"\s+.*?\s+"\$SPLIT"(?:\s+"([^"]*)")?\)', text)
    if not m:
        raise SystemExit(f"{path}: no run_dir call found")
    env["SUFFIX"] = m.group(1) or ""
    rows = []
    block = re.search(r'^ROWS=\((.*?)^\)', text, re.M | re.S)
    if not block:
        raise SystemExit(f"{path}: no ROWS=( ... ) found")
    for line in block.group(1).splitlines():
        line = line.strip()
        if line.startswith('"'):
            model, split = line.strip('"').split(",")[:2]
            rows.append((model.strip(), split.strip()))
    return env, rows


def feature_files(out_dir):
    return [
        p for p in glob.glob(os.path.join(out_dir, "results_*.json"))
        if "vllm" not in os.path.basename(p) and "shap" not in os.path.basename(p)
    ]


def run_dir(results_dir, temperature, model, split, suffix):
    """Same rule as env.sh: most feature files wins, later in sort order breaks ties."""
    plain = os.path.join(ROOT, results_dir, f"temp{temperature}_{model}_{split}{suffix}")
    candidates = sorted(glob.glob(os.path.join(ROOT, results_dir, f"*temp{temperature}_{model}_{split}{suffix}")))
    candidates = [d for d in candidates if os.path.isdir(d)]
    if not candidates:
        return plain
    return max(candidates, key=lambda d: (len(feature_files(d)), d))


def count_done(out_dir, exact=False):
    """(records on disk, number of feature files) for one run."""
    files = feature_files(out_dir)
    if exact:
        ids = set()
        for path in files:
            with open(path) as f:
                # ids sit on the line after "output": { -- json.dump(indent=2) writes
                # them one per line, so a line scan avoids parsing multi-MB records
                found = False
                for line in f:
                    if found:
                        m = re.search(r'"id":\s*(\d+)', line)
                        if m:
                            ids.add(int(m.group(1)))
                        found = False
                    elif '"output"' in line:
                        found = True
        return len(ids), len(files)

    # the file counter continues across resumed invocations, so its maximum is the
    # number of records written
    counters = []
    for path in files:
        try:
            counters.append(int(path.rsplit("_", 1)[-1].removesuffix(".json")))
        except ValueError:
            continue
    return (max(counters) if counters else 0), len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="*", default=SCRIPTS, help="run scripts to check")
    ap.add_argument("--exact", action="store_true", help="count unique example ids instead of file counters")
    ap.add_argument("--todo", action="store_true", help="only list runs that are not finished")
    args = ap.parse_args()

    todo, total = [], 0
    for script in args.scripts:
        path = os.path.join(ROOT, script)
        if not os.path.exists(path):
            raise SystemExit(f"no such script: {path}")
        env, rows = parse_script(path)
        data_name = env.get("DATA_NAME", "")
        thinking_off = env.get("THINKING", "true").lower() in ("0", "false")
        print(f"\n=== {script}  [{data_name}]")

        for model, split in dict.fromkeys(rows):
            short = os.path.basename(model)
            target = TRAIN_TARGET if split == "train" else SPLIT_TARGET.get((data_name, split))
            out_dir = run_dir(env["RESULTS_DIR"], env["TEMPERATURE"], short, split, env["SUFFIX"])
            if thinking_off:
                out_dir += "_no_thinking"
            rel = os.path.relpath(out_dir, ROOT)
            total += 1

            if target is None:
                print(f"  {'NO TARGET':<9} {short:<24} {split:<10} {rel}")
                todo.append((script, short, split, rel, "no target size known"))
                continue
            if not os.path.isdir(out_dir):
                print(f"  {'MISSING':<9} {short:<24} {split:<10} 0/{target:<6} {rel}")
                todo.append((script, short, split, rel, "directory does not exist"))
                continue

            done, n_files = count_done(out_dir, exact=args.exact)
            status = "DONE" if done >= target else "PARTIAL" if done else "EMPTY"
            note = ""
            # the counter and the file count only disagree if files were overwritten
            if not args.exact and abs(done - 5 * n_files) > 5:
                note = f"  (counter {done} vs {n_files} files -- rerun with --exact)"
            if status == "EMPTY" and glob.glob(os.path.join(out_dir, "results_vllm_*.json")):
                note = "  (vllm answers exist, HF feature stage never ran)"
            if status != "DONE":
                todo.append((script, short, split, rel, f"{done}/{target} done, {target - done} left"))
            if status == "DONE" and args.todo:
                continue
            print(f"  {status:<9} {short:<24} {split:<10} {done}/{target:<6} {rel}{note}")

    print(f"\n=== {total - len(todo)}/{total} runs complete")
    if todo:
        print("still to run:")
        for script, short, split, rel, why in todo:
            print(f"  {script:<16} {short:<24} {split:<10} {why:<28} -> {rel}")
    return 1 if todo else 0


if __name__ == "__main__":
    sys.exit(main())
