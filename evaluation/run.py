"""Scores a model (optionally with a LoRA adapter) on MBPP+, BFCL, SVAMP and IFEval.

    python -m evaluation.sources --bench bench                                   # once
    python -m evaluation.run --out runs/base --model Qwen/Qwen3-1.7B             # ~5 h on a 16 GB GPU
    python -m evaluation.run --out runs/mine --model Qwen/Qwen3-1.7B --adapter checkpoints/grpo
    python -m evaluation.run --out runs/mine_s2 ... --samples 2                   # quick look
    python -m evaluation.run --out runs/mine --regrade                            # grade stored replies again
    python -m evaluation.compare runs/base runs/mine

Writes <out>/records.jsonl (one graded reply per question and sample) and <out>/results.json.
An interrupted run continues where it stopped when started again with the same arguments; it
restarts at a batch boundary so every batch is the one an uninterrupted run would have sent.
Protocol constants: evaluation/protocol.py.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from evaluation import graders, protocol, sources
from evaluation.execution import run_programs
from evaluation.stats import interval, summarize

RECORDS, RESULTS, CONFIG = "records.jsonl", "results.json", "run_config.json"


def main():
    args = parse_args()
    at = sources.layout(args.bench)
    manifest = json.loads(at["manifest"].read_text())
    if sources.sha256(at["questions"]) != manifest["questions_sha256"]:
        raise SystemExit("bench/questions.jsonl does not match its manifest; rebuild with evaluation.sources")
    wanted = {d for b in args.benchmarks for d in protocol.BENCHMARKS[b]}
    questions = [q for q in sources.read_jsonl(at["questions"]) if q["dataset"] in wanted]
    if args.limit:
        taken, limited = Counter(), []
        for q in questions:
            taken[q["dataset"]] += 1
            if taken[q["dataset"]] <= args.limit:
                limited.append(q)
        questions = limited
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"model": args.model, "revision": args.revision,
              "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
              "benchmarks": args.benchmarks, "samples": args.samples, "seed": args.seed,
              "sampling": protocol.SAMPLING, "questions_sha256": manifest["questions_sha256"],
              "caps": {d: protocol.token_cap(d) for d in sorted(wanted)},
              "batch": protocol.BATCH, "prompt_room": protocol.PROMPT_ROOM, "limit": args.limit}
    if args.regrade:
        records = [{k: v for k, v in r.items() if k not in ("correct", "parse_error", "grading_error")}
                   for r in sources.read_jsonl(out / RECORDS)]
        config = json.loads((out / CONFIG).read_text()) if (out / CONFIG).exists() else config
    else:
        if (out / RESULTS).exists():
            raise SystemExit(f"{out} already holds a finished run (use --regrade to grade it again)")
        if (out / CONFIG).exists() and json.loads((out / CONFIG).read_text()) != config:
            raise SystemExit(f"{out} holds a partial run with different settings")
        (out / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
        records = generate(questions, args, out)
    grade(questions, records, at, args)
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    results = collect(questions, records)
    (out / RESULTS).write_text(json.dumps(dict(config, results=results), indent=2) + "\n")
    print(report(results))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--revision", help="default: the pinned revision for a known base model")
    parser.add_argument("--adapter")
    parser.add_argument("--benchmarks", default=",".join(protocol.BENCHMARKS))
    parser.add_argument("--samples", type=int, default=protocol.SAMPLES)
    parser.add_argument("--seed", type=int, default=protocol.SEED)
    parser.add_argument("--sandbox", choices=("bwrap", "rlimit"), default="bwrap")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--regrade", action="store_true")
    parser.add_argument("--limit", type=int, help="only the first N questions of each dataset (plumbing test)")
    args = parser.parse_args()
    args.benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = set(args.benchmarks) - set(protocol.BENCHMARKS)
    if unknown:
        parser.error(f"unknown benchmarks {sorted(unknown)}")
    if args.revision is None:
        args.revision = protocol.REVISIONS.get(args.model)
    return args


def generate(questions, args, out):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    eos = tokenizer.eos_token_id
    eos_ids = list(eos) if isinstance(eos, list) else [eos]
    open_id, close_id = tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
    jobs = []
    for q in questions:
        ids = tokenizer.apply_chat_template(q["messages"], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=True)
        if len(ids) > protocol.PROMPT_ROOM:
            raise SystemExit(f"question {q['id']} is {len(ids)} tokens, over the prompt room")
        for sample in range(args.samples):
            jobs.append((q, ids, sample, args.seed + 1000 * q["seed_index"] + sample))

    done = resume_point(out)
    records = sources.read_jsonl(out / RECORDS)[:done] if done else []
    if done:
        print(f"resuming after {done} of {len(jobs)} replies", flush=True)
    if done >= len(jobs):
        return records
    rank = None
    if args.adapter:
        rank = json.loads((Path(args.adapter) / "adapter_config.json").read_text())["r"]
    caps = [protocol.token_cap(d) for d in {q["dataset"] for q in questions}]
    llm = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
              max_model_len=max(caps) + protocol.PROMPT_ROOM, gpu_memory_utilization=protocol.GPU_MEMORY,
              enable_prefix_caching=False, generation_config="vllm", seed=args.seed,
              enable_lora=rank is not None, max_lora_rank=rank or 16)
    lora = LoRARequest("adapter", 1, str(Path(args.adapter).resolve())) if args.adapter else None
    with (out / RECORDS).open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
        for start in range(done, len(jobs), protocol.BATCH):
            batch = jobs[start:start + protocol.BATCH]
            params = [SamplingParams(max_tokens=protocol.token_cap(q["dataset"]), seed=seed,
                                     stop_token_ids=eos_ids, **protocol.SAMPLING)
                      for q, _, _, seed in batch]
            outputs = llm.generate([{"prompt_token_ids": ids} for _, ids, _, _ in batch], params,
                                   lora_request=lora, use_tqdm=False)
            for (q, _, sample, seed), output in zip(batch, outputs, strict=True):
                if (output.lora_request is None) != (lora is None):
                    raise RuntimeError("adapter was not applied as requested")
                tokens = list(output.outputs[0].token_ids)
                answer, reasoning, error = graders.split_reply(tokens, open_id, close_id, set(eos_ids))
                record = {"id": q["id"], "dataset": q["dataset"], "sample_index": sample, "seed": seed,
                          "output_tokens": len(tokens), "reasoning_tokens": reasoning,
                          "truncated": not (tokens and tokens[-1] in eos_ids), "boundary_error": error,
                          "text": tokenizer.decode(tokens, skip_special_tokens=True),
                          "response": tokenizer.decode(answer, skip_special_tokens=True)}
                records.append(record)
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"{start + len(batch)}/{len(jobs)}", flush=True)
    return records


def resume_point(out):
    """Complete records on disk, rounded down to a whole batch."""
    path = out / RECORDS
    if not path.exists():
        return 0
    complete = 0
    for line in path.read_text().splitlines():
        try:
            json.loads(line)
            complete += 1
        except json.JSONDecodeError:
            break
    return complete - complete % protocol.BATCH


def grade(questions, records, at, args):
    by_id = {q["id"]: q for q in questions}
    kinds = {by_id[r["id"]]["grader"] for r in records}
    check_bfcl = graders.load_bfcl_checker(at["bfcl"]) if "bfcl" in kinds else None
    grade_ifeval = graders.load_ifeval(at["ifeval"]) if "ifeval" in kinds else None
    programs = []
    if "mbpp" in kinds:
        sanitize = graders.load_sanitize(at["evalplus"])
    for key, record in enumerate(records):
        q = by_id[record["id"]]
        try:
            if q["grader"] == "numeric":
                verdict = graders.grade_svamp(record["text"], q["answer"], record["truncated"])
            elif q["grader"] == "bfcl":
                verdict = graders.grade_bfcl(q, record["response"], check_bfcl, record["truncated"])
            elif q["grader"] == "ifeval":
                verdict = grade_ifeval(q, record["response"])
            else:
                program, error = graders.mbpp_program(record["response"], q["entry_point"], sanitize,
                                                      record["truncated"])
                if error:
                    verdict = (False, error)
                else:
                    programs.append((key, q["id"], program))
                    continue
            record["correct"], record["parse_error"] = verdict
        except Exception as error:      # a grader crash leaves the row ungraded, visibly
            record.update(correct=None, parse_error=None, grading_error=f"{type(error).__name__}: {error}")
    if programs:
        print(f"running {len(programs)} MBPP+ programs on {args.workers} workers", flush=True)
        verdicts = run_programs(programs, at["evalplus"], at["mbpp"], args.workers, args.sandbox)
        for key, _, _ in programs:
            v = verdicts[key]
            if "error" in v:
                records[key].update(correct=None, parse_error=None, grading_error=v["error"])
            else:
                records[key].update(correct=v["base"] == "pass" and v["plus"] == "pass", parse_error=None)


def collect(questions, records):
    results = {}
    for dataset in sorted({q["dataset"] for q in questions}):
        rows = [r for r in records if r["dataset"] == dataset]
        errors = Counter(r["grading_error"] for r in rows if r.get("grading_error"))
        results[dataset] = dict(summarize(rows), interval=interval(rows),
                                grading_errors=dict(errors.most_common(5)))
    return results


def report(results):
    lines = [f"{'dataset':<22}{'accuracy':>10}{'95% CI':>18}{'mean tokens':>13}{'capped':>8}{'ungraded':>9}"]
    for name, r in results.items():
        lo, hi = r["interval"]["accuracy"] or (float("nan"), float("nan"))
        lines.append(f"{name:<22}{r['accuracy']:>10.2%}{f'[{lo:.1%}, {hi:.1%}]':>18}"
                     f"{r['mean_output_tokens']:>13.0f}{r['truncation_rate']:>8.1%}{r['ungraded']:>9}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
