"""GRPO from a stop-distilled adapter: reward = correct, minus a length penalty normalised per domain.

    python stopping/train_grpo.py --init checkpoints/stoppoint_mixG --out checkpoints/grpo_mixG --steps 60

Supervised stop data teaches *where* to stop from examples cut by an oracle. It cannot teach
the model *when* its own re-checking pays off: on SVAMP-like math the post-answer checks
sometimes catch an error and sometimes are wasted, and stop data either keeps them all or drops
them all (see IDEA.md, "What did not work"). Reinforcement learning with a reward for being right and short
can, because a check that saves an error earns more than a short wrong answer.

Per step: draw prompts from the training pools (GSM8K, KodCode, Glaive, constrained Dolly; never
evaluation data), sample a group of G answers per prompt with the current adapter (vLLM, the
evaluation's sampling), grade them (stopping/grading.py) and reward

    R = 1[correct] * (1 - alpha * L / m_domain)

where L is the answer's length in tokens and m_domain the domain's mean length measured on the
first batch (fixed afterwards). The length penalty is normalised per domain, not per question,
because the evaluation's metric is a ratio of sums dominated by long answers (CLAUDE.md). Wrong or
capped answers get 0 whatever their length, so the model is never paid for giving up. Advantage =
R minus the group mean (no division by the group's spread, which would inflate tiny differences).
One on-policy update per batch, so the importance ratio is 1 and the objective is the plain policy
gradient, averaged over generated tokens. No KL term: the update is a LoRA at a small learning
rate for a few dozen steps, and every 10th step is saved to choose from.

vLLM (generation) and the trained model share the GPU: vLLM takes --vllm-memory of it and loads
the newest adapter from disk each step.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.grading import grade  # noqa: E402
from evaluation.protocol import SAMPLING  # noqa: E402

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")    # e.g. BASE_MODEL=Qwen/Qwen3-1.7B
LOSS_CHUNK = 1024
POOLS = {   # domain: (files, source filter or None, max tokens)
    "math": (["data/train.jsonl"], "gsm8k", 4096),
    "code": (["data/code_train.jsonl"], None, 6144),
    "tool": (["data/tool_train_clean.jsonl"], None, 2048),
    # Wrapper prompts make the model loop after </think> (§7p); a capped reply earns 0, so RL
    # can learn to close the wrapper and stop where the loop-cut data could not be built.
    "instruction": (["data/constraint_train.jsonl", "data/wrapper_train.jsonl"], None, 4096),
}


def load_pools(weights, overrides=()):
    """Prompt rows and token cap per domain; `overrides` ("math=data/math_mix_train.jsonl") replace a
    domain's files (no source filter), e.g. by the difficulty-matched pools of stopping/mix_pools.py."""
    table = dict(POOLS)
    for item in overrides:
        domain, path = item.split("=", 1)
        table[domain] = (path.split("+"), None, POOLS[domain][2])
    pools = {}
    for domain in weights:
        paths, source, cap = table[domain]
        rows = [json.loads(l) for path in paths for l in Path(path).read_text().splitlines()]
        rows = [r for r in rows if source is None or r.get("source") == source]
        pools[domain] = (rows, cap)
    return pools


def token_logprob_sum(model, ids, prompt_length):
    """Sum of log-probabilities of the completion tokens, logits projected in chunks."""
    base = model.get_base_model()
    hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    states, targets = hidden[prompt_length - 1:-1], ids[0, prompt_length:]
    total = states.new_zeros((), dtype=torch.float32)
    for start in range(0, states.shape[0], LOSS_CHUNK):
        logits = base.lm_head(states[start:start + LOSS_CHUNK]).float()
        total = total + torch.log_softmax(logits, -1).gather(1, targets[start:start + LOSS_CHUNK, None]).sum()
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path, required=True, help="Adapter to start from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--prompts", type=int, default=32, help="Prompts per step")
    parser.add_argument("--group", type=int, default=8, help="Samples per prompt")
    parser.add_argument("--alpha", type=float, default=0.2, help="Length penalty at the domain's mean length")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weights", default="math=0.5,code=0.2,tool=0.15,instruction=0.15")
    parser.add_argument("--vllm-memory", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Continue a stopped run: adapter from <out>/latest, steps and domain means from its "
                             "log, optimizer state from <out>/optimizer.pt if present, prompt order replayed")
    parser.add_argument("--pool", action="append", default=[],
                        help="Replace a domain's prompt files, e.g. math=data/math_mix_train.jsonl (repeatable)")
    arguments = parser.parse_args()

    weights = {k: float(v) for k, v in (p.split("=") for p in arguments.weights.split(","))}
    pools = load_pools(weights, arguments.pool)
    rng = random.Random(arguments.seed)
    torch.manual_seed(arguments.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    log = (arguments.out / "log.jsonl").open("a")

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rank = json.loads((arguments.init / "adapter_config.json").read_text())["r"]
    llm = LLM(model=MODEL, gpu_memory_utilization=arguments.vllm_memory, max_model_len=6144 + 1024,
              enable_lora=True, max_lora_rank=rank, max_loras=1, max_cpu_loras=2, seed=arguments.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # Resuming: the prompt draws of the finished steps are replayed so the run continues with the prompts an
    # uninterrupted run would have drawn; the domain means are the first logged mean length per domain.
    done, minutes_before, domain_mean = 0, 0.0, {}
    if arguments.resume:
        history = [json.loads(l) for l in (arguments.out / "log.jsonl").read_text().splitlines() if l.strip()]
        done, minutes_before = len(history), history[-1]["minutes"]
        for d in weights:
            first = next((h[d]["len"] for h in history if d in h), None)
            if first is not None:
                domain_mean[d] = first
        for _ in range(done):
            for d in rng.choices(list(weights), weights=list(weights.values()), k=arguments.prompts):
                rng.choice(pools[d][0])
        print(f"resuming after step {done}; domain means {domain_mean}", flush=True)
    start_adapter = arguments.out / "latest" if arguments.resume else arguments.init
    model = PeftModel.from_pretrained(model, str(start_adapter), is_trainable=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=arguments.lr, weight_decay=0.0)
    if arguments.resume and (arguments.out / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(arguments.out / "optimizer.pt"))

    current = start_adapter
    started = time.monotonic()
    for step in range(done, arguments.steps):
        # ---- sample prompts and generate a group for each with the current adapter
        domains = rng.choices(list(weights), weights=list(weights.values()), k=arguments.prompts)
        batch = [(d, rng.choice(pools[d][0])) for d in domains]
        lora = LoRARequest(f"step{step}", step + 1, str(current.resolve()))
        prompt_ids, outputs = [], []
        for d in weights:
            part = [(i, row) for i, (dd, row) in enumerate(batch) if dd == d]
            if not part:
                continue
            ids = [tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=True) for _, row in part]
            ids = [list(x["input_ids"] if hasattr(x, "input_ids") else x) for x in ids]
            params = SamplingParams(**SAMPLING, n=arguments.group, max_tokens=pools[d][1],
                                    seed=arguments.seed * 100003 + step)
            results = llm.generate([{"prompt_token_ids": x} for x in ids], params, lora_request=lora, use_tqdm=False)
            for (i, row), x, result in zip(part, ids, results):
                for o in result.outputs:
                    prompt_ids.append(x)
                    outputs.append((i, d, row, o))
        verdicts = grade([(row, tokenizer.decode(list(o.token_ids), skip_special_tokens=True))
                          for _, _, row, o in outputs])

        # ---- rewards and group advantages
        lengths = [len(o.token_ids) for _, _, _, o in outputs]
        correct = [strict and o.finish_reason == "stop" for (strict, _), (_, _, _, o) in zip(verdicts, outputs)]
        for d in weights:
            if d not in domain_mean:
                seen = [L for L, (_, dd, _, _) in zip(lengths, outputs) if dd == d]
                if seen:
                    domain_mean[d] = sum(seen) / len(seen)
        rewards = [c * (1 - arguments.alpha * L / domain_mean[d]) for c, L, (_, d, _, _) in zip(correct, lengths, outputs)]
        groups = defaultdict(list)
        for k, (i, _, _, _) in enumerate(outputs):
            groups[i].append(k)
        advantage = [0.0] * len(outputs)
        for members in groups.values():
            mean = sum(rewards[k] for k in members) / len(members)
            for k in members:
                advantage[k] = rewards[k] - mean

        # ---- one policy-gradient update, averaged over generated tokens
        model.train()
        active = [k for k in range(len(outputs)) if abs(advantage[k]) > 1e-6]
        tokens = sum(lengths[k] + 1 for k in active) or 1
        for k in active:
            completion = list(outputs[k][3].token_ids)
            if outputs[k][3].finish_reason == "stop" and (not completion or completion[-1] != end):
                completion.append(end)
            ids = torch.tensor([prompt_ids[k] + completion], device="cuda")
            loss = -advantage[k] * token_logprob_sum(model, ids, len(prompt_ids[k])) / tokens
            loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        current = arguments.out / "latest"
        model.save_pretrained(current, safe_serialization=True)
        torch.save(optimizer.state_dict(), arguments.out / "optimizer.pt")
        if (step + 1) % 10 == 0 or step == arguments.steps - 1:
            model.save_pretrained(arguments.out / f"step{step + 1}", safe_serialization=True)

        stats = {"step": step + 1, "minutes": round(minutes_before + (time.monotonic() - started) / 60, 1),
                 "active": len(active), "samples": len(outputs)}
        for d in weights:
            ks = [k for k, (_, dd, _, _) in enumerate(outputs) if dd == d]
            if ks:
                stats[d] = {"acc": round(sum(correct[k] for k in ks) / len(ks), 3),
                            "len": round(sum(lengths[k] for k in ks) / len(ks)),
                            "capped": sum(outputs[k][3].finish_reason == "length" for k in ks),
                            # groups with both right and wrong answers: the only ones with an accuracy contrast
                            "mixed": sum(0 < sum(correct[k] for k in m) < len(m)
                                         for i, m in groups.items() if outputs[m[0]][1] == d)}
        log.write(json.dumps(stats) + "\n")
        log.flush()
        print(json.dumps(stats), flush=True)

    model.save_pretrained(arguments.out, safe_serialization=True)
    (arguments.out / "training_args.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(arguments).items()} | {"domain_mean": domain_mean}, indent=2))
    print(f"saved {arguments.out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
