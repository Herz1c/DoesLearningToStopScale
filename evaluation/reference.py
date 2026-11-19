"""Second evaluation: public long-reasoning benchmarks, under the sampling the model's vendor recommends.

    python -m evaluation.reference --prepare                                   # once: download and fix the questions
    python -m evaluation.reference --out runs/ref/base --model Qwen/Qwen3.8-27B
    python -m evaluation.reference --out runs/ref/mine --model Qwen/Qwen3.8-27B --adapter checkpoints/27b/grpo
    python -m evaluation.reference --out runs/ref/mine --regrade               # grade stored replies again
    python -m evaluation.reference --out runs/ref/mine --report                # the table again
    python -m evaluation.reference --compare runs/ref/base runs/ref/mine

The four benchmarks of evaluation/protocol.py are short tasks under a 4 096-token cap. This is the other
regime: problems on which a 27B model reasons for tens of thousands of tokens, with room to do so. The sets and
the settings are those of BottleCapAI's report on ThinkingCap-Qwen3.8-27B (2026-09-22), as far as the report
states them, so that numbers can be put next to theirs:

    aime26         MathArena/aime_2026       30 problems    answer: integer in \\boxed{}
    hmmt_feb26     MathArena/hmmt_feb_2026   33 problems    answer: expression in \\boxed{} (math-verify)
    hmmt_nov25     MathArena/hmmt_nov_2025   30 problems    same
    gpqa_diamond   Idavidrein/gpqa           198 questions  four options, shuffled once (gated: needs a login)
    livecodebench  code_generation_lite v6   175 problems   the release-6 problems; hidden tests in the sandbox
    ifbench        allenai/IFBench           300 prompts    the benchmark's own verifiers, loose, prompt level
    mmlu_pro       TIGER-Lab/MMLU-Pro        1 000 of 12 032 questions, drawn once, proportionally per category

Sampling: temperature 1.0, top-p 0.95, top-k 20; the chat template's default reasoning effort; up to 253 952
generated tokens in a 262 144-token window. A reply that does not finish is wrong. Not theirs, because the report
does not give them: the prompts (the benchmarks' usual ones, below), the answer extraction, the number of samples
(default 4 per question, MMLU-Pro 1), and their decoding speed-up (speculative decoding, which they report as
accuracy-neutral). Their harness is not public; how close this one is shows in the base model's numbers.

Reported per set, as in the report: accuracy over all replies; mean thinking tokens (tokens before </think>,
summed over replies, divided by replies) and its reduction against the base run; the "matched" reduction
(geometric mean over questions of the per-question ratio); mean total tokens; capped replies. The average over
sets is the unweighted mean of the per-set numbers.

Replies are stored whole (<out>/records.jsonl), so grading can be repeated anywhere with --regrade. A run
continues where it stopped (set by set), and raising --samples later adds samples to the same run.
"""

import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import statistics
import subprocess
import sys
import urllib.request
import zlib

from evaluation import chat

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
MAX_TOKENS = 253_952
WINDOW = 262_144
SEED = 9001
SAMPLES = {"mmlu_pro": 1}          # per question; every other set: --samples (default 4)
SETS = ("aime26", "hmmt_feb26", "hmmt_nov25", "gpqa_diamond", "livecodebench", "ifbench", "mmlu_pro")
MMLU_PRO_QUESTIONS = 1000

PINS = {     # Hub dataset revisions and the IFBench commit
    "MathArena/aime_2026": "d61f486805", "MathArena/hmmt_feb_2026": "589a53fe15", "MathArena/hmmt_nov_2025": "118dbfb45c",
    "Idavidrein/gpqa": "83022cefff", "livecodebench/code_generation_lite": "0fe84c3912", "TIGER-Lab/MMLU-Pro": "b189ec765a",
    "ifbench": "488918665d53f867fa908a293769ec6e71e74aec",        # github.com/allenai/IFBench
}
IFBENCH_FILES = ["evaluation_lib.py", "ifbench/__init__.py", "ifbench/classic_instructions.py", "ifbench/instructions.py",
                 "ifbench/instructions_registry.py", "ifbench/instructions_util.py", "data/IFBench_test.jsonl"]

# The benchmarks' usual prompts (MathArena; simple-evals multiple choice; LiveCodeBench's generic prompt).
MATH_PROMPT = "Put your final answer within \\boxed{{}}.{note}\n\n{problem}"
AIME_NOTE = "\nThe answer is an integer between 0 and 999 inclusive."
CHOICE_PROMPT = ("Answer the following multiple choice question. The last line of your response should be of the "
                 "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. "
                 "Think step by step before answering.\n\n{question}\n\n{options}")
LCB_SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
              "generate a correct Python program that matches the specification and passes all tests.")
LCB_STARTER = ("You will use the following starter code to write the solution to the problem and enclose your code "
               "within delimiters.\n```python\n{starter}\n```\n\n")
LCB_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
             "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
             "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
             "```python\n# YOUR CODE HERE\n```\n\n")
LCB_PROMPT = "### Question:\n{question}\n\n### Format: {format}### Answer: (use the provided format with backticks)\n\n"
LCB_WORKER = Path(__file__).with_name("lcb_worker.py")


def layout(bench):
    bench = Path(bench)
    return {"questions": bench / "questions.jsonl", "tests": bench / "lcb_tests.jsonl", "manifest": bench / "manifest.json",
            "ifbench": bench / "src/ifbench"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- questions

def hub(repo, name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, repo_type="dataset", revision=PINS[repo])


def math_questions(name, repo, note=""):
    import pandas as pd
    for row in pd.read_parquet(hub(repo, "data/train-00000-of-00001.parquet")).to_dict("records"):
        yield {"set": name, "id": f"{name}-{row['problem_idx']}", "grader": "math", "answer": str(row["answer"]),
               "messages": [{"role": "user", "content": MATH_PROMPT.format(note=note, problem=row["problem"].strip())}]}


def choice_question(name, key, question, options, correct, extra=None):
    letters = "ABCDEFGHIJ"[:len(options)]
    text = CHOICE_PROMPT.format(letters=letters, question=question.strip(),
                                options="\n".join(f"{letter}) {option}" for letter, option in zip(letters, options)))
    return {"set": name, "id": f"{name}-{key}", "grader": "choice", "answer": letters[correct], "letters": letters,
            "messages": [{"role": "user", "content": text}], **(extra or {})}


def gpqa_questions():
    with open(hub("Idavidrein/gpqa", "gpqa_diamond.csv"), encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(f"gpqa:{row['Record ID']}").shuffle(order)      # the options' order is fixed once per question
        yield choice_question("gpqa_diamond", row["Record ID"], row["Question"],
                              [options[i].strip() for i in order], order.index(0))


def mmlu_pro_questions():
    """1 000 questions, each category in proportion to its share of the 12 032 (largest remainders), drawn once."""
    import pandas as pd
    rows = pd.read_parquet(hub("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet")).to_dict("records")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: int(r["question_id"])):
        groups[row["category"]].append(row)
    exact = {c: MMLU_PRO_QUESTIONS * len(g) / len(rows) for c, g in groups.items()}
    take = {c: int(v) for c, v in exact.items()}
    for c in sorted(exact, key=lambda c: (take[c] - exact[c], c))[:MMLU_PRO_QUESTIONS - sum(take.values())]:
        take[c] += 1
    for category in sorted(groups):
        for row in sorted(random.Random(f"mmlu_pro:{category}").sample(groups[category], take[category]),
                          key=lambda r: int(r["question_id"])):
            yield choice_question("mmlu_pro", row["question_id"], row["question"], [str(o) for o in row["options"]],
                                  int(row["answer_index"]), {"category": category})


def lcb_questions(tests_file):
    with tests_file.open("w", encoding="utf-8") as tests:
        for row in read_jsonl(hub("livecodebench/code_generation_lite", "test6.jsonl")):
            starter = row["starter_code"]
            fmt = LCB_STARTER.format(starter=starter) if starter.strip() else LCB_STDIN
            tests.write(json.dumps({"id": f"livecodebench-{row['question_id']}", "public": row["public_test_cases"],
                                    "private": row["private_test_cases"], "metadata": row["metadata"]}) + "\n")
            yield {"set": "livecodebench", "id": f"livecodebench-{row['question_id']}", "grader": "code",
                   "difficulty": row["difficulty"], "platform": row["platform"],
                   "messages": [{"role": "system", "content": LCB_SYSTEM},
                                {"role": "user", "content": LCB_PROMPT.format(question=row["question_content"], format=fmt)}]}


def ifbench_questions(root):
    base = f"https://raw.githubusercontent.com/allenai/IFBench/{PINS['ifbench']}"
    for name in IFBENCH_FILES:
        target = root / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{base}/{name}", timeout=120) as reply:
                target.write_bytes(reply.read())
    for row in read_jsonl(root / "data/IFBench_test.jsonl"):
        yield {"set": "ifbench", "id": f"ifbench-{row['key']}", "grader": "ifbench", "key": row["key"],
               "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"], "original_prompt": row["prompt"],
               "messages": [{"role": "user", "content": row["prompt"]}]}


def prepare(bench):
    """Downloads the sets and writes the question file every run reads. A set that cannot be fetched (GPQA
    needs a Hugging Face login that has accepted its terms) is left out, loudly; run --prepare again to add it."""
    at = layout(bench)
    at["questions"].parent.mkdir(parents=True, exist_ok=True)
    have = read_jsonl(at["questions"]) if at["questions"].exists() else []
    builders = {"aime26": lambda: math_questions("aime26", "MathArena/aime_2026", AIME_NOTE),
                "hmmt_feb26": lambda: math_questions("hmmt_feb26", "MathArena/hmmt_feb_2026"),
                "hmmt_nov25": lambda: math_questions("hmmt_nov25", "MathArena/hmmt_nov_2025"),
                "gpqa_diamond": gpqa_questions, "livecodebench": lambda: lcb_questions(at["tests"]),
                "ifbench": lambda: ifbench_questions(at["ifbench"]), "mmlu_pro": mmlu_pro_questions}
    questions = []
    for index, name in enumerate(SETS):
        rows = [q for q in have if q["set"] == name]
        if not rows:
            try:
                rows = list(builders[name]())
            except Exception as error:
                print(f"NOT AVAILABLE: {name} ({type(error).__name__}: {str(error)[:160]})", flush=True)
                continue
            for k, q in enumerate(rows):
                q["seed_index"] = 100_000 * index + k
        questions += rows
        print(f"{name:14s} {len(rows):5d} questions", flush=True)
    at["questions"].write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    digest = hashlib.sha256(at["questions"].read_bytes()).hexdigest()
    at["manifest"].write_text(json.dumps({"questions_sha256": digest, "pins": PINS, "sets": dict(Counter(q["set"] for q in questions)),
                                          "sampling": SAMPLING, "max_tokens": MAX_TOKENS}, indent=2) + "\n")
    print(f"{len(questions)} questions, sha256 {digest[:12]}")
    try:
        import nltk
        for package in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"):
            nltk.download(package, quiet=True)
    except Exception as error:
        print(f"note: NLTK data not fetched ({error}); IFBench grading needs it")


# ---------------------------------------------------------------- generation

def generate(questions, args, out):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.convert_tokens_to_ids("<|im_end|>")}
    _, close_id = chat.think_ids(tokenizer)
    records = read_jsonl(out / "records.jsonl") if (out / "records.jsonl").exists() else []
    done = {(r["id"], r["sample"]) for r in records}
    jobs = defaultdict(list)
    for q in questions:
        for sample in range(SAMPLES.get(q["set"], args.samples)):
            if (q["id"], sample) not in done:
                jobs[q["set"]].append((q, sample))
    if not jobs:
        return records
    rank = json.loads((Path(args.adapter) / "adapter_config.json").read_text())["r"] if args.adapter else None
    llm = LLM(model=args.model, max_model_len=min(WINDOW, args.max_tokens + 8192), gpu_memory_utilization=0.85,
              enable_prefix_caching=False, generation_config="vllm", seed=SEED, max_num_seqs=args.max_seqs,
              enable_lora=rank is not None, max_lora_rank=rank or 16, **chat.engine_options(args.model))
    lora = LoRARequest("adapter", 1, str(Path(args.adapter).resolve())) if args.adapter else None
    with (out / "records.jsonl").open("a", encoding="utf-8") as stream:
        for name in SETS:
            part = jobs.get(name)
            if not part:
                continue
            print(f"{name}: generating {len(part)} replies", flush=True)
            prompts = [{"prompt_token_ids": chat.prompt_ids(tokenizer, q["messages"])} for q, _ in part]
            params = [SamplingParams(max_tokens=args.max_tokens, seed=SEED + 1000 * q["seed_index"] + sample,
                                     stop_token_ids=sorted(eos_ids), **SAMPLING) for q, sample in part]
            outputs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
            for (q, sample), output in zip(part, outputs, strict=True):
                if (output.lora_request is None) != (lora is None):
                    raise RuntimeError("adapter was not applied as requested")
                tokens = list(output.outputs[0].token_ids)
                finished = bool(tokens) and tokens[-1] in eos_ids
                closed = close_id in tokens
                thinking = tokens.index(close_id) if closed else len(tokens)
                answer = [t for t in tokens[thinking + 1:] if t not in eos_ids] if closed else []
                record = {"set": name, "id": q["id"], "sample": sample, "tokens": len(tokens), "thinking_tokens": thinking,
                          "answer_tokens": len(answer), "truncated": not finished, "closed": closed,
                          "text": chat.whole_reply(tokenizer, tokenizer.decode(tokens, skip_special_tokens=True)),
                          "response": tokenizer.decode(answer, skip_special_tokens=True)}
                records.append(record)
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"{name}: done", flush=True)
    return records


# ---------------------------------------------------------------- grading

def boxed(text):
    """Content of the last \\boxed{...}, braces matched."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    start = text.find("{", start)
    if start < 0:
        return None
    depth = 0
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:position]
    return None


def grade_math(gold, response):
    answer = boxed(response)
    if answer is None:
        return False, "no_boxed_answer"
    plain = lambda s: re.sub(r"\\!|\\,|\\ |\s|\$|,(?=\d{3})", "", s).strip(".")
    if plain(answer) == plain(gold):
        return True, None
    if re.fullmatch(r"-?\d+", plain(gold)) and re.fullmatch(r"-?\d+(\.0+)?", plain(answer)):
        return int(float(plain(answer))) == int(plain(gold)), None
    from math_verify import parse, verify
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${answer}$"))), None
    except Exception as error:      # noqa: BLE001 -- a parser failure is a wrong answer, with its reason
        return False, f"unparsed: {type(error).__name__}"


def grade_choice(question, response):
    found = re.findall(r"Answer:\s*\**\s*\$?\(?([A-J])\b", response)
    if not found:
        return False, "no_answer_line"
    return found[-1] == question["answer"], None


def load_ifbench(root):
    """grade(question, response): the benchmark's loose check, all constraints of the prompt."""
    import langdetect
    langdetect.DetectorFactory.seed = 0
    sys.path.insert(0, str(root))
    try:
        import evaluation_lib
    finally:
        sys.path.remove(str(root))

    def grade(question, response):
        kwargs = [{k: v for k, v in item.items() if v is not None} for item in question["kwargs"]]
        example = evaluation_lib.InputExample(key=question["key"], instruction_id_list=question["instruction_id_list"],
                                              prompt=question["original_prompt"], kwargs=kwargs)
        random.seed(f"ifbench:{question['id']}")
        result = evaluation_lib.test_instruction_following_loose(example, {question["original_prompt"]: response})
        return bool(result.follow_all_instructions), None
    return grade


def code_block(response):
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", response, re.S)
    return blocks[-1] if blocks else None


def decode_tests(row):
    tests = json.loads(row["public"])
    try:
        tests += json.loads(row["private"])
    except json.JSONDecodeError:
        tests += json.loads(pickle.loads(zlib.decompress(base64.b64decode(row["private"].encode("utf-8")))))
    return tests, (json.loads(row["metadata"]) if row["metadata"] else {}).get("func_name")


def run_code(items, tests_file, workers):
    """[(key, question id, program)] -> {key: (passed, error)}; programs run in the code sandbox, one worker
    process per share, every test with its own time limit."""
    from concurrent.futures import ThreadPoolExecutor
    from stopping.code_tasks import sandbox_command
    tests = {row["id"]: row for row in read_jsonl(tests_file)}
    shares = [items[i::workers] for i in range(workers)]

    def run(share):
        payload = []
        for key, question_id, program in share:
            cases, name = decode_tests(tests[question_id])
            payload.append(json.dumps({"key": key, "code": program, "tests": cases, "func_name": name}))
        finished = subprocess.run(sandbox_command(LCB_WORKER), input="\n".join(payload) + "\n", text=True,
                                  capture_output=True)
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
    return merged


def grade(questions, records, at, workers, skip_code=False):
    by_id = {q["id"]: q for q in questions}
    ifbench = None
    programs = []
    for key, record in enumerate(records):
        q = by_id.get(record["id"])
        if q is None:
            continue
        if record["truncated"] or not record["closed"]:
            record["correct"], record["error"] = False, "truncated" if record["truncated"] else "thinking_not_closed"
            continue
        response = record["response"]
        try:
            if q["grader"] == "math":
                verdict = grade_math(q["answer"], response)
            elif q["grader"] == "choice":
                verdict = grade_choice(q, response)
            elif q["grader"] == "ifbench":
                ifbench = ifbench or load_ifbench(at["ifbench"])
                verdict = ifbench(q, response)
            else:
                program = code_block(response)
                if program is None:
                    verdict = (False, "no_code_block")
                else:
                    programs.append((key, q["id"], program))
                    continue
            record["correct"], record["error"] = verdict
        except Exception as error:      # noqa: BLE001 -- a grader crash leaves the reply ungraded, visibly
            record["correct"], record["error"] = None, f"grader: {type(error).__name__}: {str(error)[:120]}"
    if programs:
        from stopping.code_tasks import self_test
        problem = "not run here (--no-code)" if skip_code else self_test()
        if problem:
            print(f"{len(programs)} programs NOT run ({problem[:120]}): grade this run with --regrade where a sandbox works", flush=True)
            for key, _, _ in programs:
                records[key]["correct"], records[key]["error"] = None, "not run: no sandbox"
        else:
            print(f"running {len(programs)} programs on {workers} workers", flush=True)
            for key, (passed, error) in run_code(programs, at["tests"], workers).items():
                records[key]["correct"], records[key]["error"] = passed, error


# ---------------------------------------------------------------- reporting

def summarize(rows):
    graded = [r for r in rows if r.get("correct") is not None]
    return {"replies": len(rows), "questions": len({r["id"] for r in rows}), "ungraded": len(rows) - len(graded),
            "accuracy": sum(r["correct"] for r in graded) / len(graded) if graded else None,
            "mean_thinking": sum(r["thinking_tokens"] for r in rows) / len(rows),
            "median_thinking": statistics.median(r["thinking_tokens"] for r in rows),
            "mean_answer": sum(r["answer_tokens"] for r in rows) / len(rows),
            "mean_total": sum(r["tokens"] for r in rows) / len(rows),
            "median_total": statistics.median(r["tokens"] for r in rows),
            "truncated": sum(r["truncated"] for r in rows) / len(rows),
            "errors": dict(Counter(r["error"] for r in rows if r.get("error")).most_common(6))}


def report(records):
    lines = [f"{'set':<14}{'replies':>8}{'accuracy':>10}{'thinking':>10}{'median':>8}{'answer':>8}{'total':>8}{'capped':>8}{'ungraded':>9}"]
    results = {}
    for name in SETS:
        rows = [r for r in records if r["set"] == name]
        if not rows:
            continue
        s = results[name] = summarize(rows)
        accuracy = "not graded" if s["accuracy"] is None else f"{s['accuracy']:.2%}"
        lines.append(f"{name:<14}{s['replies']:>8}{accuracy:>10}{s['mean_thinking']:>10.0f}{s['median_thinking']:>8.0f}"
                     f"{s['mean_answer']:>8.0f}{s['mean_total']:>8.0f}{s['truncated']:>8.1%}{s['ungraded']:>9}")
    return results, "\n".join(lines)


def compare(base_dir, new_dir, resamples=2000):
    """Per set and on average: accuracy change, reduction of mean thinking tokens, matched reduction, with
    intervals from a paired bootstrap over questions (the samples both runs have)."""
    load = lambda d: {(r["id"], r["sample"]): r for r in read_jsonl(Path(d) / "records.jsonl")}
    base, new = load(base_dir), load(new_dir)
    rng = random.Random(0)
    print(f"base  {base_dir}\nnew   {new_dir}\n")
    print(f"{'set':<14}{'questions':>10}{'acc base':>10}{'acc new':>9}{'change':>23}{'thinking base':>15}{'new':>8}{'reduction':>24}{'matched':>9}{'capped':>14}")
    rows_out = []
    for name in SETS:
        keys = [k for k in base if k in new and base[k]["set"] == name and base[k].get("correct") is not None
                and new[k].get("correct") is not None]
        if not keys:
            continue
        by_q = defaultdict(list)
        for k in keys:
            by_q[k[0]].append(k)
        ids = sorted(by_q)

        def stats(sel):
            ks = [k for q in sel for k in by_q[q]]
            acc = lambda d: sum(d[k]["correct"] for k in ks) / len(ks)
            think = lambda d: sum(d[k]["thinking_tokens"] for k in ks) / len(ks)
            return acc(base), acc(new), think(base), think(new)
        ab, an, tb, tn = stats(ids)
        ratios = [sum(new[k]["thinking_tokens"] for k in by_q[q]) / sum(base[k]["thinking_tokens"] for k in by_q[q])
                  for q in ids if sum(base[k]["thinking_tokens"] for k in by_q[q]) and sum(new[k]["thinking_tokens"] for k in by_q[q])]
        matched = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) - 1 if ratios else float("nan")
        draws = [stats([rng.choice(ids) for _ in ids]) for _ in range(resamples)]
        lo_hi = lambda values: (sorted(values)[int(0.025 * resamples)], sorted(values)[int(0.975 * resamples)])
        d_lo, d_hi = lo_hi([100 * (d[1] - d[0]) for d in draws])
        r_lo, r_hi = lo_hi([d[3] / d[2] - 1 for d in draws])
        capped = lambda d: sum(d[k]["truncated"] for k in keys) / len(keys)
        print(f"{name:<14}{len(ids):>10}{ab:>10.2%}{an:>9.2%}{f'{100 * (an - ab):+.2f} pp [{d_lo:+.2f}, {d_hi:+.2f}]':>23}"
              f"{tb:>15.0f}{tn:>8.0f}{f'{tn / tb - 1:+.1%} [{r_lo:+.1%}, {r_hi:+.1%}]':>24}{matched:>+9.1%}"
              f"{f'{capped(base):.1%} -> {capped(new):.1%}':>14}")
        rows_out.append((ab, an, tn / tb - 1, matched))
    if rows_out:
        n = len(rows_out)
        print(f"\naverage over {n} sets (unweighted): accuracy {sum(r[0] for r in rows_out) / n:.2%} -> {sum(r[1] for r in rows_out) / n:.2%} "
              f"({100 * sum(r[1] - r[0] for r in rows_out) / n:+.2f} pp), mean thinking {sum(r[2] for r in rows_out) / n:+.1%}, "
              f"matched {sum(r[3] for r in rows_out) / n:+.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default="bench_reference")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3.8-27B"))
    parser.add_argument("--adapter")
    parser.add_argument("--sets", default=",".join(SETS))
    parser.add_argument("--samples", type=int, default=4, help="Replies per question (MMLU-Pro: always 1)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-seqs", type=int, default=128, help="vLLM's max_num_seqs")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="Only the first N questions of each set (plumbing test)")
    parser.add_argument("--no-code", action="store_true", help="Do not run programs here; --regrade them elsewhere")
    parser.add_argument("--regrade", action="store_true")
    parser.add_argument("--report", action="store_true", help="Print the table of a finished run")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.report:
        return print(report(read_jsonl(args.out / "records.jsonl"))[1])
    at = layout(args.bench)
    if args.prepare:
        return prepare(args.bench)
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
    questions = [q for q in read_jsonl(at["questions"]) if q["set"] in wanted]
    if args.limit:
        seen = Counter()
        questions = [q for q in questions if not (seen.update([q["set"]]) or seen[q["set"]] > args.limit)]
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"model": args.model, "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
              "sampling": SAMPLING, "max_tokens": args.max_tokens, "samples": args.samples, "limit": args.limit,
              "questions_sha256": json.loads(at["manifest"].read_text())["questions_sha256"]}
    if args.regrade:
        records = [{k: v for k, v in r.items() if k not in ("correct", "error")} for r in read_jsonl(args.out / "records.jsonl")]
    else:
        (args.out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        records = generate(questions, args, args.out)
    grade(read_jsonl(at["questions"]), records, at, args.workers, args.no_code)
    (args.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    results, table = report(records)
    (args.out / "results.json").write_text(json.dumps(dict(config, results=results), indent=2) + "\n")
    print(table)


if __name__ == "__main__":
    main()
    if "--out" in sys.argv and "--regrade" not in sys.argv and "--report" not in sys.argv:
        chat.leave()
