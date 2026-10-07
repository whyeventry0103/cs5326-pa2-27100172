# %% [markdown]
# Kaggle Benchmarks task (A/B order swapped for swap-consistency): blind pairwise judging of the PA2 comparisons on Kaggle's model proxy.
#
# The attached private dataset holds the *_pairs.jsonl files and judge_pairwise_kaggle.py, so this
# runs exactly the same prompt, batching, retry, and resume logic as the CLI. Push and run with
#   kaggle b t push pa2-pairwise-judge -f scripts/kaggle_judge_task.py -d <owner>/pa2-judge-pairs --wait
#   kaggle b t run pa2-pairwise-judge -m gemini-3.6-flash --wait

# %%
import glob
import os
import sys

import kaggle_benchmarks as kbench

# PA2_DATA_ROOT lets the task be validated locally (`python kaggle_judge_task.py`) before pushing.
DATA_ROOT = os.environ.get("PA2_DATA_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("PA2_OUT_DIR", "/kaggle/working/judgments")
hits = glob.glob(os.path.join(DATA_ROOT, "**", "judge_pairwise_kaggle.py"), recursive=True)
assert hits, "attach the pa2-judge-pairs-swapped dataset"
DATA_DIR = os.path.dirname(hits[0])
sys.path.insert(0, DATA_DIR)
import judge_pairwise_kaggle as judge_lib  # noqa: E402


# %%
@kbench.task(name="pa2-pairwise-judge-swapped")
def pa2_pairwise_judge_swapped(llm) -> dict:
    name = getattr(llm, "name", None) or os.environ.get("LLM_DEFAULT", "kaggle-default")
    judge = judge_lib.KaggleJudge(str(name), llm=llm)
    os.makedirs(OUT_DIR, exist_ok=True)
    # Task creation executes this notebook once on Kaggle's default model; only the intended judge
    # model does the full ~1.4k-pair run, anything else is a 5-pair smoke test.
    full = any(m in str(name) for m in os.environ.get("PA2_FULL_JUDGES", "gemini-3.6-flash").split(","))
    pair_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_pairs.jsonl")))
    if not full:
        smoke = os.path.join(OUT_DIR, "smoke_pairs.jsonl")
        with open(pair_files[0]) as src, open(smoke, "w") as dst:
            dst.writelines(src.readlines()[:5])
        pair_files = [smoke]
    print(f"judge={name} full_run={full} files={len(pair_files)}", flush=True)
    results = {}
    for pairs_path in pair_files:
        key = os.path.basename(pairs_path)[: -len("_pairs.jsonl")]
        out = os.path.join(OUT_DIR, f"{key}_judged.jsonl")
        print(f"=== {key}", flush=True)
        summary = judge_lib.judge_file(judge, pairs_path, out, batch_size=5, sleep_seconds=5, max_retries=3)
        results[key] = {"summary": summary, "judged": judge_lib.read_jsonl(out)}
        print(key, {k: summary[k] for k in ("wins", "losses", "ties", "unknown", "failed")}, flush=True)
    return results


pa2_pairwise_judge_swapped.run(kbench.llm)
