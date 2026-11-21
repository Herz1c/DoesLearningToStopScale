"""From where in its own thinking does the model already have the answer, and does it keep it?

    python stopping/readouts.py runs/27b_long_cal/math                  # forced answers at up to 12 boundaries
    python stopping/readouts.py runs/27b_long_cal/math --report         # the table, from what is written

For every rollout of stopping/long_rollouts.py this takes up to 12 paragraph boundaries spread evenly over the
thinking (the rollout's own end included), closes the thinking there and samples two answers at each. A boundary
is *right* when both answers are right ("knows", stopping/grading.py). The rollout's *settled point* is the
earliest of the chosen boundaries from which every later one is right as well: the answer is known and stays
known. This is the profile the stop search of the earlier studies assumed (it searched for the earliest right
boundary by bisection); on reasoning of tens of thousands of tokens it is measured instead of assumed.

The report gives, per part of the pool and per length of the thinking, how much of the thinking lies after the
settled point - an estimate of what stopping can save at all - next to the share lying after the *first* right
boundary. With 12 boundaries the settled point is found late rather than early, so the estimate is a lower one.
Rollouts are read from <dir>/rollouts*.jsonl; results go to <dir>/readouts*.jsonl. Resumable by rollout.

    python stopping/readouts.py runs/27b_long_cal --tasks code_io       # only the rollouts of these tasks
    python stopping/readouts.py runs/27b_long_cal --regrade             # programs graded again, engine not loaded

The forced answers of program tasks are kept in the file ("answers"), because a judge on a busy machine cannot be
trusted (2026-11-21: time limits on right programs while the engine was loaded): --regrade runs the kept programs
and the programs of the rollouts themselves again and rewrites the verdicts; the earlier files stay as *.before.
"""

import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import chat  # noqa: E402
from stopping.grading import grade  # noqa: E402

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.8-27B")
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
ANSWER_TOKENS = {"math_boxed": 1024, "math": 512, "code_io": 2048, "code": 1024, "multiple_choice": 256,
                 "constraint": 1024, "tool": 256}
POINTS, SAMPLES = 12, 2
BUCKETS = ((0, 2000, "< 2k"), (2000, 8000, "2-8k"), (8000, 32000, "8-32k"), (32000, 10 ** 9, "> 32k"))


def read(directory, pattern):
    rows = []
    for path in sorted(Path(directory).glob(pattern)):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip().endswith("}")]
    return rows


def spread(cuts, points=POINTS):
    """At most `points` of the boundaries, evenly spread, the last one always among them."""
    if len(cuts) <= points:
        return list(cuts)
    return [cuts[round(i * (len(cuts) - 1) / (points - 1))] for i in range(points)]


def settled(cuts):
    """(settled cut or None, first right cut or None) of a rollout's readouts."""
    right = [all(c["right"]) for c in cuts]
    first = next((c["cut"] for c, r in zip(cuts, right) if r), None)
    point = None
    for c, r in zip(reversed(cuts), reversed(right)):
        if not r:
            break
        point = c["cut"]
    return point, first


def regrade(directory):
    """Program tasks only: the rollouts' own answers and the kept forced answers through the judge once more."""
    changed = collections.Counter()
    tests = {}
    for path in sorted(Path(directory).glob("rollouts*.jsonl")):
        rows = read(directory, path.name)
        mine = [r for r in rows if r["task"] == "code_io"]
        tests.update({(r["question_id"], r["sample"]): r for r in mine})
        for r, (strict, known) in zip(mine, grade([(r, r["text"]) for r in mine])):
            changed[("rollouts", bool(r["knows"]), bool(known))] += 1
            r["correct"], r["knows"] = strict, known
        rewrite(path, rows)
    for path in sorted(Path(directory).glob("readouts*.jsonl")):
        rows = read(directory, path.name)
        cells = [(r, c, text) for r in rows if r["task"] == "code_io" for c in r["cuts"] for text in c.get("answers", [])]
        grades = grade([(tests[(r["question_id"], r["sample"])], "</think>" + text) for r, c, text in cells])
        fresh = collections.defaultdict(list)
        for (r, c, _), (_, known) in zip(cells, grades):
            fresh[(r["question_id"], r["sample"], c["cut"])].append(bool(known))
        for r in rows:
            if r["task"] != "code_io":
                continue
            source = tests[(r["question_id"], r["sample"])]
            r["correct"], r["knows"] = source["correct"], source["knows"]
            for c in r["cuts"]:
                new = fresh.get((r["question_id"], r["sample"], c["cut"]))
                if new is None:
                    changed[("readouts without kept answers", "-", "-")] += len(c["right"])
                    continue
                for before, after in zip(c["right"], new):
                    changed[("readouts", before, after)] += 1
                c["right"] = new
        rewrite(path, rows)
    print("regrade (file, right before, right now): count")
    for key, count in sorted(changed.items(), key=str):
        print(f"  {key}: {count}")


def rewrite(path, rows):
    before = path.with_name(path.name + ".before")
    if not before.exists():
        os.replace(path, before)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(temporary, path)


def report(directory):
    rows = read(directory, "readouts*.jsonl")
    if not rows:
        return print(f"{directory}: no readouts")
    print(f"{directory}: {len(rows)} rollouts with readouts")
    print(f"{'part':16} {'thinking':8} {'rollouts':>8} {'knows':>6} {'mean think':>10} {'own end right':>13} "
          f"{'settled early':>13} {'after settled':>13} {'after first right':>17}")
    groups = collections.defaultdict(list)
    for r in rows:
        bucket = next(name for low, high, name in BUCKETS if low <= r["thinking_tokens"] < high)
        for key in ((r.get("part", "-"), "all"), ("ALL", "all"), ("ALL", bucket)):
            groups[key].append(r)
    for (part, bucket), group in sorted(groups.items(), key=lambda kv: (kv[0][0] == "ALL", kv[0])):
        usable = [r for r in group if r["closed"] and r["cuts"] and all(r["cuts"][-1]["right"])]
        think = sum(r["thinking_tokens"] for r in usable) or 1
        points = [settled(r["cuts"]) for r in usable]
        after_settled = sum(r["thinking_tokens"] - p for r, (p, _) in zip(usable, points)) / think
        after_first = sum(r["thinking_tokens"] - f for r, (_, f) in zip(usable, points)) / think
        early = sum(p < r["cuts"][-1]["cut"] for r, (p, _) in zip(usable, points)) / max(1, len(usable))
        print(f"{part:16} {bucket:8} {len(group):8} {100 * sum(r['knows'] for r in group) / len(group):5.1f}% "
              f"{sum(r['thinking_tokens'] for r in group) / len(group):10.0f} {100 * len(usable) / len(group):12.1f}% "
              f"{100 * early:12.1f}% {100 * after_settled:12.1f}% {100 * after_first:16.1f}%")
    print("own end right = closed rollouts whose forced answer at their own end is right twice (the base for the "
          "last three columns);\nsettled early = of those, a settled point before the own end; after settled / "
          "after first right = share of their thinking tokens lying after that point.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--regrade", action="store_true", help="Program tasks: judge the kept answers again")
    parser.add_argument("--tasks", help="Comma-separated tasks; only their rollouts get readouts")
    parser.add_argument("--judge-later", action="store_true",
                        help="Program tasks: keep the answers, mark them wrong for now, run --regrade afterwards")
    parser.add_argument("--shard", default="0/1", help="i/n: the questions whose id is i modulo n")
    parser.add_argument("--wave", type=int, default=150, help="Rollouts per engine call")
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261121)
    arguments = parser.parse_args()
    if arguments.report:
        return report(arguments.directory)
    if arguments.regrade:
        return regrade(arguments.directory)

    index, count = (int(x) for x in arguments.shard.split("/"))
    rollouts = [r for r in read(arguments.directory, "rollouts*.jsonl") if r["question_id"] % count == index
                and (not arguments.tasks or r["task"] in arguments.tasks.split(","))]
    path = arguments.directory / (f"readouts.{index}of{count}.jsonl" if count > 1 else "readouts.jsonl")
    done = {(r["question_id"], r["sample"]) for r in read(arguments.directory, path.name)}
    todo = [r for r in rollouts if (r["question_id"], r["sample"]) not in done]
    print(f"{path}: {len(rollouts) - len(todo)} of {len(rollouts)} rollouts done, {len(todo)} to go", flush=True)
    if not todo:
        return

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from stopping.find_stop_points import boundaries, forced_ids      # the same boundaries and closing as the stop search
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    longest = max(len(chat.prompt_ids(tokenizer, r["prompt"])) + r["thinking_tokens"] for r in todo)
    llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.90, seed=arguments.seed,
              max_model_len=longest + max(ANSWER_TOKENS.values()) + 64, max_num_seqs=arguments.max_seqs,
              **chat.engine_options(MODEL))

    if any(r["task"] == "code_io" for r in todo):
        from stopping.long_tasks import judge_timing
        print("with the engine loaded -", end=" ", flush=True)
        judge_timing()
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(todo), arguments.wave):
            wave = todo[start:start + arguments.wave]
            jobs = []      # consecutive boundaries of one rollout share their prefix: kept next to each other
            for r in wave:
                prompt = chat.prompt_ids(tokenizer, r["prompt"])
                for cut in spread(boundaries(tokenizer, r)):
                    jobs.append((r, cut, forced_ids(prompt, r, cut)))
            params = [SamplingParams(**SAMPLING, n=SAMPLES, max_tokens=ANSWER_TOKENS.get(r["task"], 1024),
                                     seed=arguments.seed + 7 * r["question_id"] + cut) for r, cut, _ in jobs]
            results = llm.generate([{"prompt_token_ids": ids} for _, _, ids in jobs], params, use_tqdm=False) if jobs else []
            answers = [(r, cut, o) for (r, cut, _), result in zip(jobs, results) for o in result.outputs]
            later = {k for k, (r, _, _) in enumerate(answers) if arguments.judge_later and r["task"] == "code_io"}
            now = iter(grade([(r, "</think>" + o.text) for k, (r, cut, o) in enumerate(answers) if k not in later]))
            grades = [(False, False) if k in later else next(now) for k in range(len(answers))]
            cuts = collections.defaultdict(lambda: collections.defaultdict(lambda: {"right": [], "answer_tokens": []}))
            for (r, cut, o), (_, known) in zip(answers, grades):
                cell = cuts[(r["question_id"], r["sample"])][cut]
                cell["right"].append(bool(known))
                cell["answer_tokens"].append(len(o.token_ids))
                if r["task"] == "code_io":
                    cell.setdefault("answers", []).append(o.text)
            for r in wave:
                mine = cuts.get((r["question_id"], r["sample"]), {})
                stream.write(json.dumps({
                    **{k: r[k] for k in ("question_id", "sample", "source", "task", "tokens", "thinking_tokens",
                                         "closed", "finished", "correct", "knows")},
                    "part": r.get("part", "-"),
                    "cuts": [dict(cut=cut, **mine[cut]) for cut in sorted(mine)]}) + "\n")
            stream.flush()
            minutes = (time.monotonic() - started) / 60
            print(f"{start + len(wave)}/{len(todo)} rollouts, {len(jobs)} boundaries, {minutes:.0f} min", flush=True)


if __name__ == "__main__":
    main()
    chat.leave()
