"""Rollouts of the base model (or an adapter) on long-reasoning tasks: the vendor's sampling, a long cap.

    python stopping/long_rollouts.py --data data/long_cal_math.jsonl --out runs/27b_long_cal/math
    python stopping/long_rollouts.py --data ... --out ... --shard 0/2      # half of the questions (one card)

Differs from stopping/generate_rollouts.py (kept as it is for the earlier studies) in what long reasoning needs:
temperature 1.0 / top-p 0.95 / top-k 20 (the sampling Qwen recommends for thinking and the reference evaluation
uses), a cap of 65 536 tokens instead of 4 096, questions taken in file order (question_id = line number), a
question's seed fixed by its id so that shards on two cards give what one run would, and all questions of a
wave submitted together so that a few very long replies do not hold up short ones.

Rows carry the generated token ids, the grades (stopping/grading.py: strict and "knows") and the row's `part`.
Resumable: a question is written when all its samples are graded; running again skips written questions.
"""

import argparse
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
CARRY = ("tests", "entry_point", "functions", "variant", "part", "name", "rating", "contest", "domain")


def shard_of(text):
    index, count = (int(x) for x in text.split("/"))
    return index, count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2, help="Rollouts per question")
    parser.add_argument("--max-tokens", type=int, default=65536, help="Generation cap per rollout")
    parser.add_argument("--shard", default="0/1", help="i/n: the questions whose line number is i modulo n")
    parser.add_argument("--limit", type=int, help="Only the first N questions of the file (tests)")
    parser.add_argument("--wave", type=int, default=400, help="Questions per engine call; a stop loses one wave")
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--judge-later", action="store_true",
                        help="Programs are not run here (marked wrong); stopping/readouts.py --regrade judges them "
                             "once the engine is gone")
    arguments = parser.parse_args()

    index, count = shard_of(arguments.shard)
    rows = [json.loads(line) for line in arguments.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [dict(row, question_id=k) for k, row in enumerate(rows[:arguments.limit]) if k % count == index]
    arguments.out.mkdir(parents=True, exist_ok=True)
    path = arguments.out / (f"rollouts.{index}of{count}.jsonl" if count > 1 else "rollouts.jsonl")
    done = set()
    if path.exists():
        kept = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip().endswith("}")]
        per = {}
        for r in kept:
            per[r["question_id"]] = per.get(r["question_id"], 0) + 1
        done = {q for q, n in per.items() if n == arguments.samples}
        path.write_text("".join(json.dumps(r) + "\n" for r in kept if r["question_id"] in done), encoding="utf-8")
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{path}: {len(questions) - len(todo)} of {len(questions)} questions done, {len(todo)} to go", flush=True)
    if not todo:
        return

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    _, close_id = chat.think_ids(tokenizer)
    rank = json.loads((arguments.adapter / "adapter_config.json").read_text())["r"] if arguments.adapter else None
    llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.90, seed=arguments.seed,
              max_model_len=arguments.max_tokens + 8192, max_num_seqs=arguments.max_seqs,
              enable_lora=rank is not None, max_lora_rank=rank or 16, **chat.engine_options(MODEL))
    lora = LoRARequest("candidate", 1, str(arguments.adapter.resolve())) if arguments.adapter else None

    started, tokens = time.monotonic(), 0
    with path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(todo), arguments.wave):
            wave = todo[start:start + arguments.wave]
            prompts = [{"prompt_token_ids": chat.prompt_ids(tokenizer, q["prompt"])} for q in wave]
            params = [SamplingParams(**SAMPLING, n=arguments.samples, max_tokens=arguments.max_tokens,
                                     seed=arguments.seed + q["question_id"]) for q in wave]
            records = []
            for q, result in zip(wave, llm.generate(prompts, params, lora_request=lora, use_tqdm=False)):
                for k, output in enumerate(result.outputs):
                    ids = list(output.token_ids)
                    closed = close_id in ids
                    records.append({
                        "question_id": q["question_id"], "sample": k, "source": q["source"], "task": q["task"],
                        "answer": q["answer"], "prompt": q["prompt"], **{key: q[key] for key in CARRY if key in q},
                        "tokens": len(ids), "closed": closed,
                        "thinking_tokens": ids.index(close_id) + 1 if closed else len(ids),
                        "finished": output.finish_reason == "stop",
                        "text": tokenizer.decode(ids, skip_special_tokens=True), "token_ids": ids})
            later = {k for k, r in enumerate(records) if arguments.judge_later and r["task"] == "code_io"}
            now = iter(grade([(r, r["text"]) for k, r in enumerate(records) if k not in later]))
            for k, record in enumerate(records):
                record["correct"], record["knows"] = (False, False) if k in later else next(now)
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            tokens += sum(r["tokens"] for r in records)
            minutes = (time.monotonic() - started) / 60
            print(f"{start + len(wave)}/{len(todo)} questions, {minutes:.0f} min, {tokens / 1e6:.2f} M tokens "
                  f"({tokens / max(minutes, 0.01) * 60 / 1e6:.1f} M/h), knows {sum(r['knows'] for r in records)}/{len(records)}",
                  flush=True)


if __name__ == "__main__":
    main()
    chat.leave()     # stops the vLLM engine and exits; its own teardown can hang after outputs are written
