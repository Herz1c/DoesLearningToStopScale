"""Training tasks for long reasoning: harder sources, one format per domain, graders, and the pool builder.

    python stopping/long_tasks.py --sources ~/sources_long --out data --count 250      # calibration pools
    python stopping/long_tasks.py --judge-timing                                       # how fast the code judge runs here

Sources (all permissively licensed; none is an evaluation benchmark or one of its sources):

    math        DeepMath-103K (MIT), five difficulty bands; HARP (MIT) short-answer contest problems, AMC only
                (every AIME contest is left out: AIME 2026 is an evaluation set, and so the whole contest stays out)
    code        CodeContests (CC BY 4.0), Codeforces problems with a rating, collected up to 2021-07, five rating
                bands; kept only when one of the dataset's own Python 3 solutions passes the chosen tests here
    choice      MedQA USMLE, four options (CC BY 4.0 on the Hub copy used), steps 1 and 2&3; a few easy questions
                from the existing training file (ARC-Easy, CommonsenseQA)
    constraint  the existing own-constraint prompts on Dolly, without the emoji constraint (IFBench's test set has
                an emoji-per-sentence constraint; the neighbour is dropped)
    tool        the existing Glaive prompts

A candidate that shares 15% or more of its 8-grams with the evaluation questions given in --questions is dropped.
Row format is the one of the other pools (prompt, answer, task, source, part); two new task names:

    math_boxed  "Put your final answer within \\boxed{}."; right when the last \\boxed{} after the thinking equals
                the reference (string, integer, or symbolic comparison - evaluation/reference.py, grade_math)
    code_io     a Python 3 program reading standard input; right when the last code block passes the row's tests
                (evaluation/lcb_worker.py in the code sandbox); strict additionally wants exactly one code block
"""

import argparse
import collections
import glob
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATH_SUFFIX = "\n\nPut your final answer within \\boxed{}."
CODE_PREFIX = ("Write a Python 3 program that solves the following problem. The program reads its input from standard "
               "input and writes its answer to standard output. Return the program in one Python code block.\n\n")
CHOICE_SUFFIX = "\n\nAnswer with the label of the correct option. Put it on the last line, like: Final answer: B"
MATH_BANDS = (("dm_3-4", 3.0, 4.5), ("dm_5", 5.0, 5.5), ("dm_6", 6.0, 6.5), ("dm_7", 7.0, 7.5), ("dm_8-9", 8.0, 9.5))
CODE_BANDS = (("cf_800-1200", 800, 1200), ("cf_1300-1600", 1300, 1600), ("cf_1700-2000", 1700, 2000),
              ("cf_2100-2400", 2100, 2400), ("cf_2500+", 2500, 3600))
MAX_TESTS, TEST_BYTES, PROBLEM_BYTES = 20, 20_000, 60_000
LCB_WORKER = str(Path(__file__).resolve().parents[1] / "evaluation" / "lcb_worker.py")
BLOCK = re.compile(r"```(?:python|py|Python|python3)?[ \t]*\n(.*?)```", re.S)


# ---- grading ---------------------------------------------------------------------------------------------------

def judge_workers():
    """Parallel judge processes: JUDGE_WORKERS, or the cores this job may really use (on a cluster os.cpu_count()
    is the whole node), at most 12."""
    if os.environ.get("JUDGE_WORKERS"):
        return max(1, int(os.environ["JUDGE_WORKERS"]))
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(12, cores))


def run_io(items, workers=None, retry=True):
    """[(key, program, tests)] -> {key: (passed, error)}; tests are [{"input", "output"}], run as stdin tests.
    A "time limit" or a missing verdict is not believed from a busy machine: those programs run once more, one
    at a time. (2026-11-21: with the engine still loaded the cluster failed 304 of 359 right programs.)"""
    from concurrent.futures import ThreadPoolExecutor
    from stopping.code_tasks import sandbox_command
    workers = workers or judge_workers()
    shares = [items[i::workers] for i in range(workers)]

    def run(share):
        payload = "".join(json.dumps({"key": key, "code": program, "func_name": None,
                                      "tests": [dict(t, testtype="stdin") for t in tests]}) + "\n"
                          for key, program, tests in share)
        finished = subprocess.run(sandbox_command(LCB_WORKER), input=payload, text=True, capture_output=True)
        verdicts = {}
        for line in finished.stdout.splitlines():
            if line.strip():
                verdict = json.loads(line)
                verdicts[verdict["key"]] = (verdict["passed"], verdict["error"])
        for key, _, _ in share:
            verdicts.setdefault(key, (None, f"sandbox: no verdict ({finished.stderr.strip()[-200:]})"))
        return verdicts
    merged = {}
    with ThreadPoolExecutor(workers) as pool:
        for part in pool.map(run, [s for s in shares if s]):
            merged.update(part)
    doubtful = [item for item in items if not merged[item[0]][0]
                and ("time limit" in str(merged[item[0]][1]) or "no verdict" in str(merged[item[0]][1]))]
    if retry and workers > 1 and doubtful:
        again = run_io(doubtful, workers=1, retry=False)
        changed = sum(bool(again[key][0]) for key, _, _ in doubtful)
        print(f"judge: {len(doubtful)} of {len(items)} programs timed out or gave no verdict with {workers} workers; "
              f"alone {changed} of them pass", flush=True)
        merged.update(again)
    return merged


def grade_boxed(row, text):
    """(strict, knows) for a math_boxed row; strict also wants the thinking closed exactly once."""
    from evaluation.reference import grade_math
    right = grade_math(str(row["answer"]), text.split("</think>")[-1])[0]
    return right and text.count("</think>") <= 1, right


def grade_io(pairs):
    """[(row, text)] -> [(strict, knows)] for code_io rows, in one sandboxed batch."""
    jobs, blocks = [], []
    for k, (row, text) in enumerate(pairs):
        found = BLOCK.findall(text.split("</think>")[-1])
        blocks.append(found)
        if found:
            jobs.append((k, found[-1], row["tests"]))
    verdicts = run_io(jobs) if jobs else {}
    return [(bool(verdicts.get(k, (False,))[0]) and len(blocks[k]) == 1, bool(verdicts.get(k, (False,))[0]))
            for k in range(len(pairs))]


def judge_timing(count=24):
    """Seconds per trivial program through the judge, alone and with every worker busy. A program is one Python
    start plus one test; anything near the judge's 6 s limit explains "time limit" verdicts on right programs."""
    program, tests = "print(sum(map(int, input().split())))\n", [{"input": "2 3\n", "output": "5\n"}]
    for workers in (1, judge_workers()):
        started = time.monotonic()
        verdicts = run_io([(k, program, tests * 3) for k in range(count)], workers=workers, retry=False)
        seconds = time.monotonic() - started
        passed = sum(bool(v[0]) for v in verdicts.values())
        errors = collections.Counter(str(v[1]) for v in verdicts.values() if not v[0]).most_common(2)
        print(f"judge timing: {workers:2} workers, {count} programs x 3 tests: {seconds:6.1f} s total, "
              f"{seconds * workers / (count * 3):5.2f} s per test, {passed}/{count} passed {errors if errors else ''}", flush=True)


# ---- pools -----------------------------------------------------------------------------------------------------

def grams(text, n=8):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlap_filter(paths):
    """drop(text) -> True when 15% or more of the text's 8-grams occur in the evaluation questions."""
    seen = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            content = row.get("messages") or row.get("prompt") or row.get("question") or ""
            if isinstance(content, list):
                content = " ".join(m["content"] for m in content)
            seen |= grams(str(content))

    def drop(text):
        own = grams(text)
        return bool(own) and len(own & seen) / len(own) >= 0.15
    return drop


def user(text):
    return [{"role": "user", "content": text}]


def math_rows(sources, count, rng, drop):
    import pandas as pd
    frames = [pd.read_parquet(f, columns=["question", "final_answer", "difficulty"])
              for f in sorted(glob.glob(f"{sources}/DeepMath-103K/data/*.parquet"))]
    table = pd.concat(frames).drop_duplicates("question")
    rows, per = [], count // (len(MATH_BANDS) + 1)
    for name, low, high in MATH_BANDS:
        part = table[(table["difficulty"] >= low) & (table["difficulty"] <= high)]
        picked = []
        for q, a in rng.sample(list(zip(part["question"], part["final_answer"])), len(part)):
            a = str(a).strip()
            # yes/no answers are right half the time by guessing, which would fool a stop search
            if a.lower() in ("yes", "no", "true", "false") or "\\text" in a or len(q) > 1500 or drop(q):
                continue
            picked.append({"prompt": user(q.strip() + MATH_SUFFIX), "answer": a, "task": "math_boxed",
                           "source": "deepmath", "part": name})
            if len(picked) == per:
                break
        rows += picked
    archive = zipfile.ZipFile(f"{sources}/HARP.jsonl.zip")
    harp = [json.loads(line) for line in archive.open(archive.namelist()[0])]
    plain = re.compile(r"[-+0-9./ ]+|[-+0-9 ]*\\d?frac\{\d+\}\{\d+\}|[-+0-9 ]*\\sqrt\{?\d+\}?")
    picked = []
    for r in rng.sample(harp, len(harp)):
        answer = str(r["answer"]).strip().strip("$").strip()
        if (r["contest"].startswith("AIME") or int(r["level"]) < 3 or str(r["multiple_choice_only"]) == "True"
                or not plain.fullmatch(answer) or drop(r["problem"])):
            continue
        picked.append({"prompt": user(r["problem"].strip() + MATH_SUFFIX), "answer": answer, "task": "math_boxed",
                       "source": "harp", "part": "harp_amc", "contest": f"{r['contest']} {r['year']}"})
        if len(picked) == count - len(rows):
            break
    return rows + picked


def code_candidates(sources, rng, drop):
    """Codeforces problems with a rating, a Python 3 solution and enough small tests; one dict each."""
    import pandas as pd
    found = []
    for path in sorted(glob.glob(f"{sources}/code_contests/data/*.parquet")):
        for r in pd.read_parquet(path).to_dict("records"):
            if (r["source"] != 2 or not r["cf_rating"] or r["is_description_translated"] or r["input_file"]
                    or r["output_file"] or len(r["description"]) > 6000 or drop(r["description"])):
                continue
            solutions = [str(s) for s, lang in zip(r["solutions"]["solution"], r["solutions"]["language"]) if lang == 3]
            solutions = sorted({s for s in solutions if len(s) < 4000}, key=len)
            tests, size = [], 0
            for group in ("public_tests", "private_tests", "generated_tests"):
                for i, o in zip(r[group]["input"], r[group]["output"]):
                    if len(tests) < MAX_TESTS and len(i) + len(o) <= TEST_BYTES and size + len(i) + len(o) <= PROBLEM_BYTES:
                        tests.append({"input": str(i), "output": str(o)})
                        size += len(i) + len(o)
            if solutions and len(tests) >= 6:
                found.append({"name": r["name"], "rating": int(r["cf_rating"]), "description": r["description"],
                              "tests": tests, "solutions": rng.sample(solutions, min(3, len(solutions)))})
    return found


def code_rows(sources, count, rng, drop):
    candidates = code_candidates(sources, rng, drop)
    rng.shuffle(candidates)
    candidates = list({c["description"].strip(): c for c in candidates}.values())      # the dataset repeats some problems
    per, chosen = count // len(CODE_BANDS), []
    for name, low, high in CODE_BANDS:
        band = [c for c in candidates if low <= c["rating"] <= high][:3 * per]
        # a problem is usable only when a known-right program passes its tests in this judge (answers that are
        # not unique, or programs too slow in Python, would otherwise make every right reply look wrong)
        verdicts = run_io([(f"{i}:{j}", s, c["tests"]) for i, c in enumerate(band) for j, s in enumerate(c["solutions"])])
        usable = [c for i, c in enumerate(band) if any(verdicts.get(f"{i}:{j}", (False,))[0] for j in range(len(c["solutions"])))]
        print(f"  code {name}: {len(band)} candidates, {len(usable)} pass with a reference solution", flush=True)
        chosen += [{"prompt": user(CODE_PREFIX + c["description"].strip()), "answer": "", "task": "code_io",
                    "source": "codecontests", "part": name, "tests": c["tests"], "name": c["name"], "rating": c["rating"]}
                   for c in usable[:per]]
    return chosen


def choice_rows(sources, count, rng, drop, train_file):
    medqa = [json.loads(line) for line in open(f"{sources}/MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl", encoding="utf-8")]
    rows, easy = [], count // 5
    for step, name in (("step1", "medqa_step1"), ("step2&3", "medqa_step23")):
        picked = []
        for r in rng.sample(medqa, len(medqa)):
            if r["meta_info"] != step or drop(r["question"]):
                continue
            options = "\n".join(f"{label}. {text}" for label, text in sorted(r["options"].items()))
            picked.append({"prompt": user(f"{r['question'].strip()}\n\n{options}{CHOICE_SUFFIX}"), "answer": r["answer_idx"],
                           "task": "multiple_choice", "source": "medqa", "part": name})
            if len(picked) == (count - easy) // 2:
                break
        rows += picked
    if train_file and Path(train_file).exists():
        old = [json.loads(line) for line in Path(train_file).read_text().splitlines()]
        old = [r for r in old if r["task"] == "multiple_choice" and not drop(r["prompt"][0]["content"])]
        rows += [dict(r, part="easy_choice") for r in rng.sample(old, easy)]
    return rows


def old_rows(path, count, rng, drop, keep=lambda r: True):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    rows = [r for r in rows if keep(r) and not drop(r["prompt"][-1]["content"])]
    return rng.sample(rows, min(count, len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default=os.path.expanduser("~/qwen-scaling/sources_long"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--name", default="long_cal", help="File prefix: <out>/<name>_<domain>.jsonl")
    parser.add_argument("--count", type=int, default=250, help="Questions per domain (tool: 100)")
    parser.add_argument("--questions", nargs="*", default=[], help="Evaluation question files to stay clear of")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domains", default="math,code,choice,constraint,tool")
    parser.add_argument("--judge-timing", action="store_true")
    parser.add_argument("--counts", help="Per-domain counts, e.g. math=550,code=370,tool=100 (default: --count, tool 100)")
    parser.add_argument("--exclude", nargs="*", default=[], type=Path,
                        help="Pool files whose questions are not drawn again (validation and later pools)")
    parser.add_argument("--all", action="store_true", help="Also write <name>_all.jsonl: every domain, shuffled")
    parser.add_argument("--constraint-file", default="pool27b_constraint.jsonl",
                        help="Constrained prompts in --out to draw the constraint domain from")
    arguments = parser.parse_args()
    if arguments.judge_timing:
        return judge_timing()
    drop = overlap_filter(arguments.questions)
    data = arguments.out
    used = {json.dumps(json.loads(line)["prompt"]) for path in arguments.exclude
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    counts = {d: (100 if d == "tool" else arguments.count) for d in ("math", "code", "choice", "constraint", "tool")}
    counts.update({k: int(v) for k, v in (item.split("=") for item in (arguments.counts or "").split(",") if item)})
    extra = len(used)        # drawn on top, so that enough is left once the used questions are taken out
    builders = {
        "math": lambda rng, n: math_rows(arguments.sources, n, rng, drop),
        "code": lambda rng, n: code_rows(arguments.sources, n, rng, drop),
        "choice": lambda rng, n: choice_rows(arguments.sources, n, rng, drop, data / "train.jsonl"),
        "constraint": lambda rng, n: old_rows(data / arguments.constraint_file, n, rng, drop,
                                              lambda r: all(c["kind"] != "emojis" for c in r["answer"])),
        "tool": lambda rng, n: old_rows(data / "pool27b_tool.jsonl", n, rng, drop),
    }
    everything = []
    for k, domain in enumerate(arguments.domains.split(",")):
        rows = builders[domain](random.Random(arguments.seed * 100 + k), counts[domain] + (extra if used else 0))
        rows = [r for r in rows if json.dumps(r["prompt"]) not in used]
        parts = collections.defaultdict(list)       # trimmed back to the wanted count, the parts kept in balance
        for r in rows:
            parts[r.get("part", "-")].append(r)
        kept = []
        while len(kept) < counts[domain] and any(parts.values()):
            for name in sorted(parts):
                if parts[name] and len(kept) < counts[domain]:
                    kept.append(parts[name].pop(0))
        rows = [dict(r, domain=domain) for r in kept]
        assert len({json.dumps(r["prompt"]) for r in rows}) == len(rows), f"{domain}: repeated prompts"
        path = data / f"{arguments.name}_{domain}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{path}: {len(rows)} rows (wanted {counts[domain]}) {dict(collections.Counter(r.get('part', '-') for r in rows))}", flush=True)
        everything += rows
    if arguments.all:
        random.Random(arguments.seed + 20261123).shuffle(everything)      # shares and waves then mix the domains
        path = data / f"{arguments.name}_all.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in everything), encoding="utf-8")
        print(f"{path}: {len(everything)} rows", flush=True)


if __name__ == "__main__":
    main()
