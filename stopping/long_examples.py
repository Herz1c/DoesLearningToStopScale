"""Training rows of the long-reasoning track, for both objectives, from rollouts and their readouts.

    python stopping/long_examples.py runs/27b_long_cal --out runs/27b_long_pilot --sample      # engine: answers at the cuts
    python stopping/long_examples.py runs/27b_long_cal --out runs/27b_long_pilot --build       # no engine: grade, write
    (several run directories may be given; question ids are then kept apart by directory)

A rollout is *usable* when it closed its thinking and the two answers forced at its own end are both right. Its
*settled point* is the earliest of the read-out boundaries from which every later one is right twice
(stopping/readouts.py). Rollouts that are not usable never teach stopping.

trace.jsonl - "truncated traces", the objective of the earlier studies (stopping/train_long.py --objective trace):
    the model's own thinking up to the settled point, </think>, and a right answer sampled there (the shortest of
    four that passes the strict grader). A usable rollout that settles only at its own end is kept whole when its
    own answer is strictly right: those rows say "not yet". No give-up rows, no fixed budget.

decision.jsonl - "stop decision only" (--objective decision): every rollout whole, as the base model wrote it,
    with the positions where the next token should be </think>: every paragraph boundary at or after the settled
    point of a usable rollout ("stop_positions"), and the boundaries where it must go on as the base
    model did ("hold_positions"): those up to the last read-out that was still wrong, and every boundary of a
    rollout that is not usable. Boundaries between the last wrong read-out and the settled one are not known
    either way: they are neither. Everywhere the trainer holds the model to the base model's distribution.

--sample needs the engine and only writes <out>/answers.jsonl (resumable); --build grades those answers with the
engine gone (the code judge is not trusted next to a loaded engine) and writes the two files.
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
from stopping.readouts import ANSWER_TOKENS, SAMPLING, read, settled  # noqa: E402

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.8-27B")
SAMPLES = 4


def load(directories):
    """Rollouts joined with their readouts; key = (directory index, question, sample)."""
    rows = []
    for d, directory in enumerate(directories):
        readouts = {(r["question_id"], r["sample"]): r for r in read(directory, "readouts*.jsonl")}
        for r in read(directory, "rollouts*.jsonl"):
            cuts = readouts.get((r["question_id"], r["sample"]), {}).get("cuts", [])
            r["key"] = f"{d}:{r['question_id']}:{r['sample']}"
            r["usable"] = bool(r["closed"] and cuts and all(cuts[-1]["right"]))
            r["settled"], r["last_cut"] = (settled(cuts)[0], cuts[-1]["cut"]) if r["usable"] else (None, None)
            r["last_wrong"] = max((c["cut"] for c in cuts if not all(c["right"])), default=0)
            rows.append(r)
    return rows


def sample(rows, out, arguments):
    path = out / "answers.jsonl"
    done = {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [r for r in rows if r["usable"] and r["settled"] < r["last_cut"] and r["key"] not in done]
    print(f"{path}: {len(done)} done, {len(todo)} cuts to sample", flush=True)
    if not todo:
        return
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from stopping.find_stop_points import forced_ids
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = {r["key"]: chat.prompt_ids(tokenizer, r["prompt"]) for r in todo}
    longest = max(len(prompts[r["key"]]) + r["settled"] for r in todo)
    llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.90, seed=arguments.seed,
              max_model_len=longest + max(ANSWER_TOKENS.values()) + 64, max_num_seqs=arguments.max_seqs,
              **chat.engine_options(MODEL))
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(todo), arguments.wave):
            wave = todo[start:start + arguments.wave]
            params = [SamplingParams(**SAMPLING, n=SAMPLES, max_tokens=ANSWER_TOKENS.get(r["task"], 1024),
                                     seed=arguments.seed + 11 * r["question_id"] + r["sample"]) for r in wave]
            inputs = [{"prompt_token_ids": forced_ids(prompts[r["key"]], r, r["settled"])} for r in wave]
            for r, result in zip(wave, llm.generate(inputs, params, use_tqdm=False)):
                stream.write(json.dumps({"key": r["key"], "cut": r["settled"], "answers": [
                    {"text": o.text, "token_ids": list(o.token_ids), "finished": o.finish_reason == "stop"}
                    for o in result.outputs]}) + "\n")
            stream.flush()
            print(f"{start + len(wave)}/{len(todo)} cuts, {(time.monotonic() - started) / 60:.0f} min", flush=True)


def build(rows, out, arguments):
    from transformers import AutoTokenizer
    from stopping.find_stop_points import DOUBLE_NEWLINE, THINK_CLOSE, boundaries
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    path = out / "answers.jsonl"
    sampled = {a["key"]: a for a in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else {}
    by_key = {r["key"]: r for r in rows}
    pairs = [(by_key[key], answer) for key, a in sampled.items() if key in by_key for answer in a["answers"]]
    verdicts = collections.defaultdict(list)
    for (r, answer), (strict, _) in zip(pairs, grade([(r, "</think>" + a["text"]) for r, a in pairs])):
        if strict and answer["finished"]:
            verdicts[r["key"]].append(answer)

    kinds, trace, decision = collections.Counter(), [], []
    for r in rows:
        ids, common = r["token_ids"], {k: r.get(k) for k in ("key", "task", "part", "source")}
        marks = boundaries(tokenizer, r)
        stops = [b for b in marks if b >= r["settled"]] if r["usable"] else []
        holds = [b for b in marks if b <= r["last_wrong"]] if r["usable"] else marks
        decision.append({**common, "prompt": r["prompt"], "completion_ids": ids, "stop_positions": stops,
                         "hold_positions": holds,
                         "usable": r["usable"], "finished": r["finished"]})
        if not r["usable"]:
            kind = "not usable (ends wrong or never closed)"
        elif r["settled"] < r["last_cut"] and verdicts.get(r["key"]) and sampled[r["key"]]["cut"] == r["settled"]:
            answer = min(verdicts[r["key"]], key=lambda a: len(a["token_ids"]))
            trace.append({**common, "prompt": r["prompt"], "kind": "settled early", "cut": r["settled"],
                          "thinking_tokens": r["thinking_tokens"],
                          "completion_ids": ids[:r["settled"]] + [THINK_CLOSE, DOUBLE_NEWLINE] + answer["token_ids"]})
            kind = "settled early"
        elif r["correct"] and r["finished"]:
            trace.append({**common, "prompt": r["prompt"], "kind": "whole", "cut": r["thinking_tokens"],
                          "thinking_tokens": r["thinking_tokens"], "completion_ids": ids})
            kind = "whole (settles at its own end)" if r["settled"] >= r["last_cut"] else "whole (no strict answer at the cut)"
        else:
            kind = "dropped (no strict answer at the cut, own answer not strict)"
        kinds[kind] += 1
    for name, data in (("trace", trace), ("decision", decision)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    print(f"{len(rows)} rollouts")
    for kind, count in kinds.most_common():
        print(f"  {count:6}  {kind}")
    early = [t for t in trace if t["kind"] == "settled early"]
    kept = sum(t["cut"] for t in trace) / max(1, sum(t["thinking_tokens"] for t in trace))
    print(f"trace.jsonl: {len(trace)} rows ({len(early)} cut early), thinking kept {100 * kept:.1f}% of these rollouts', "
          f"{sum(len(t['completion_ids']) for t in trace) / 1e6:.2f} M completion tokens")
    with_stop = [d for d in decision if d["stop_positions"]]
    print(f"decision.jsonl: {len(decision)} rows, {len(with_stop)} with stop positions "
          f"({sum(len(d['stop_positions']) for d in decision)} positions), "
          f"{sum(len(d['hold_positions']) for d in decision)} hold positions, "
          f"{sum(len(d['completion_ids']) for d in decision) / 1e6:.2f} M completion tokens")
    by_task = collections.Counter((t["task"], t["kind"]) for t in trace)
    print("trace rows by task:", dict(sorted(by_task.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--wave", type=int, default=400)
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261123)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    rows = load(arguments.directories)
    if arguments.sample:
        sample(rows, arguments.out, arguments)
        chat.leave()
    if arguments.build:
        build(rows, arguments.out, arguments)


if __name__ == "__main__":
    main()
