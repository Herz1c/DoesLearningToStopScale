"""Probe of a new base model before a study is planned for it: does it load, think, train and take an adapter?

    BASE_MODEL=Qwen/Qwen3.8-27B python scripts/probe_model.py template --out runs/27b_probe
    BASE_MODEL=...              python scripts/probe_model.py generate --out runs/27b_probe --text-only
    BASE_MODEL=...              python scripts/probe_model.py train    --out runs/27b_probe
    BASE_MODEL=...              python scripts/probe_model.py generate --out runs/27b_probe --text-only \
                                    --adapter runs/27b_probe/adapter --name lora --per-pool 4 --memory 0.45

Nothing here is an experiment; every part answers a plumbing question and writes <out>/<name>.json:

- template  how the chat template opens the assistant turn (Qwen3 lets the model write "<think>"; later
            templates put it into the prompt), whether <think> and </think> are single tokens, and what
            apply_chat_template(tokenize=True) returns in this transformers version;
- generate  vLLM: load time, memory, tokens per second, and for a few questions of each training pool the
            length, the share cut off at the cap, the shape of the thinking block and the grade; with
            --adapter the same through a LoRA adapter (does vLLM accept one for this architecture?) and
            with --memory a smaller share of the card (can it sit next to a model being trained?);
- train     Transformers + PEFT: which class loads the model, which linear layers exist and which of the
            recipe's LoRA targets match them, then optimiser steps on one sequence of growing length up to
            the recipe's longest: seconds and peak memory, until the card is full. Saves the adapter.

Run by chains/probe_27b.sh, one process per part.
"""

import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]     # as stopping/train_lora.py
POOLS = {"math": "data/pool4b_math.jsonl", "code": "data/pool4b_code.jsonl", "tool": "data/pool4b_tool.jsonl",
         "constraint": "data/pool4b_constraint.jsonl"}


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return [[int(x) for x in line.split(",")] for line in out]
    except Exception as error:      # noqa: BLE001 - a probe reports, it does not stop
        return str(error)


def versions():
    out = {"model": MODEL, "python": sys.version.split()[0]}
    for name in ("torch", "transformers", "vllm", "peft"):
        try:
            out[name] = __import__(name).__version__
        except Exception as error:      # noqa: BLE001
            out[name] = f"missing: {error}"
    return out


def questions(per_pool):
    rows = []
    for name, path in POOLS.items():
        if Path(path).exists():
            with open(path) as stream:
                rows += [json.loads(line) for _, line in zip(range(per_pool), stream)]
    return rows


def write(arguments, report):
    arguments.out.mkdir(parents=True, exist_ok=True)
    report = versions() | report
    (arguments.out / f"{arguments.name}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


def template(arguments):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    report = {"tokenizer": type(tokenizer).__name__, "vocabulary": len(tokenizer), "prompts": {}}
    for tag in ("<think>", "</think>"):
        ids = tokenizer.encode(tag, add_special_tokens=False)
        report[tag] = {"ids": ids, "single_token": len(ids) == 1,
                       "special": tag in getattr(tokenizer, "all_special_tokens", [])}
    for row in questions(1):
        text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
        report["prompts"][row["task"]] = {"characters": len(text), "tokens": len(tokenizer(text)["input_ids"]),
                                          "tail": text[-80:], "think_in_prompt": text.rstrip().endswith("<think>")}
        tokenized = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=True)
        report["tokenize_true_returns"] = type(tokenized).__name__
    write(arguments, report)


def generate(arguments):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from evaluation.protocol import SAMPLING
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rank = json.loads((arguments.adapter / "adapter_config.json").read_text())["r"] if arguments.adapter else None
    options = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if arguments.text_only else {}
    if rank:
        options |= {"enable_lora": True, "max_lora_rank": rank}
    start = time.time()
    llm = LLM(model=MODEL, gpu_memory_utilization=arguments.memory, max_model_len=arguments.max_tokens + 4096,
              seed=9001, **options)
    report = {"options": options, "memory_share": arguments.memory, "load_seconds": round(time.time() - start),
              "gpu_after_load_mib": gpu_memory()}
    lora = None
    if rank:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("probe", 1, str(arguments.adapter.resolve()))
    rows = questions(arguments.per_pool)
    prompts = [tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True) for r in rows]
    params = SamplingParams(**SAMPLING, n=arguments.samples, max_tokens=arguments.max_tokens, seed=9001)
    start = time.time()
    results = llm.generate(prompts, params, lora_request=lora)
    seconds = time.time() - start
    records = []
    for row, result in zip(rows, results):
        for k, o in enumerate(result.outputs):
            text = tokenizer.decode(list(o.token_ids), skip_special_tokens=True)
            records.append({"task": row["task"], "sample": k, "tokens": len(o.token_ids),
                            "capped": o.finish_reason == "length", "text": text, "row": row})
    total = sum(r["tokens"] for r in records)
    report |= {"questions": len(rows), "samples": arguments.samples, "responses": len(records), "tokens": total,
               "seconds": round(seconds, 1), "tokens_per_second": round(total / max(seconds, 1e-9)),
               "sampling": SAMPLING, "max_tokens": arguments.max_tokens}
    try:
        from stopping.grading import grade
        verdicts = grade([(r["row"], r["text"]) for r in records])
    except Exception:      # noqa: BLE001
        report["grading_error"] = traceback.format_exc()[-1500:]
        verdicts = [(None, None)] * len(records)
    report["by_task"] = {}
    for task in sorted({r["task"] for r in records}):
        part = [(r, v) for r, v in zip(records, verdicts) if r["task"] == task]
        closes = collections.Counter(r["text"].count("</think>") for r, _ in part)
        report["by_task"][task] = {
            "responses": len(part), "mean_tokens": round(sum(r["tokens"] for r, _ in part) / len(part)),
            "capped": sum(r["capped"] for r, _ in part),
            "closing_tags_per_response": dict(sorted(closes.items())),
            "opens_with_think_tag": sum(r["text"].lstrip().startswith("<think>") for r, _ in part),
            "strict": None if part[0][1][0] is None else sum(bool(v[0]) for _, v in part),
            "knows": None if part[0][1][1] is None else sum(bool(v[1]) for _, v in part)}
    report["example"] = {r["task"]: r["text"][:300] + " [...] " + r["text"][-300:] for r in records if r["sample"] == 0}
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / f"{arguments.name}.jsonl").open("w") as stream:
        for r, (strict, knows) in zip(records, verdicts):
            r.pop("row")
            stream.write(json.dumps(r | {"strict": strict, "knows": knows}) + "\n")
    write(arguments, report)


def train(arguments):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    report = {}
    start = time.time()
    model = None
    for loader in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, loader).from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
            report["loader"] = loader
            break
        except Exception:      # noqa: BLE001
            report[f"{loader}_error"] = traceback.format_exc()[-1500:]
    if model is None:
        write(arguments, report)
        return
    report |= {"class": type(model).__name__, "load_seconds": round(time.time() - start),
               "parameters": sum(p.numel() for p in model.parameters()),
               "weights_gib": round(torch.cuda.memory_allocated() / 2**30, 1)}
    census = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            entry = census.setdefault(name.split(".")[-1], {"count": 0, "parameters": 0, "example": name,
                                                            "shape": list(module.weight.shape)})
            entry["count"] += 1
            entry["parameters"] += module.weight.numel()
    report["linear_layers"] = dict(sorted(census.items(), key=lambda item: -item[1]["parameters"]))
    found = [t for t in TARGETS if t in census]
    linear_total = sum(e["parameters"] for e in census.values())
    report |= {"recipe_targets_found": found, "recipe_targets_missing": [t for t in TARGETS if t not in census],
               "share_of_linear_parameters_covered": round(sum(census[t]["parameters"] for t in found) / linear_total, 3)}
    if not found:
        write(arguments, report)
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        report["gradient_checkpointing"] = True
    except Exception as error:      # noqa: BLE001
        report["gradient_checkpointing"] = str(error)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=found,
                                             task_type="CAUSAL_LM"))
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    vocabulary = model.get_input_embeddings().weight.shape[0]
    generator = torch.Generator().manual_seed(0)
    report["steps"] = []
    for length in arguments.lengths:
        # As stopping/train_lora.py: one sequence per backward pass (a batch is accumulated gradients), the
        # loss projected in chunks; if this model is not built like that, the plain loss, and the report says so.
        entry = {"length": length}
        try:
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, min(vocabulary, 100000), (1, length), generator=generator).to("cuda")
            times = []
            for _ in range(2):          # the second step is the honest timing
                start = time.time()
                try:
                    from stopping.train_lora import chunked_loss
                    loss = chunked_loss(model, ids, 64)
                    entry["loss_path"] = "chunked, as train_lora.py"
                except (AttributeError, TypeError) as error:
                    entry["loss_path"] = f"plain (chunked failed: {error})"
                    loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(round(time.time() - start, 2))
            entry |= {"seconds": times, "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                      "loss": round(float(loss), 3)}
        except torch.OutOfMemoryError:
            entry["out_of_memory"] = True
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except Exception:      # noqa: BLE001
            entry["error"] = traceback.format_exc()[-1500:]
        report["steps"].append(entry)
        print(entry, flush=True)
        if "seconds" not in entry:
            break
    try:
        model.save_pretrained(str(arguments.out / "adapter"))
        report["adapter"] = sorted(p.name for p in (arguments.out / "adapter").iterdir())
    except Exception:      # noqa: BLE001
        report["adapter_error"] = traceback.format_exc()[-1500:]
    report["gpu_at_end_mib"] = gpu_memory()
    write(arguments, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=("template", "generate", "train"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", help="Name of the report file (default: the part)")
    parser.add_argument("--per-pool", type=int, default=16, help="Questions taken from each training pool")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--memory", type=float, default=0.9, help="vLLM's share of the card")
    parser.add_argument("--text-only", action="store_true", help="Tell vLLM to load no image or video inputs")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")], default=[1024, 2048, 4096, 7400],
                        help="Sequence lengths of the training steps, shortest first (the recipe's longest is 7400)")
    arguments = parser.parse_args()
    arguments.name = arguments.name or arguments.part
    try:
        {"template": template, "generate": generate, "train": train}[arguments.part](arguments)
    except Exception:      # noqa: BLE001
        write(arguments, {"failed": traceback.format_exc()[-3000:]})


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
