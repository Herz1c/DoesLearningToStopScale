"""LoRA training on long reasoning (sequences of tens of thousands of tokens), with two objectives.

    python stopping/train_long.py --data runs/27b_long_pilot/trace.jsonl --out checkpoints/27b_long/pilot_trace
    python stopping/train_long.py --data runs/27b_long_pilot/decision.jsonl --out checkpoints/27b_long/pilot_decision

The objective follows the file (stopping/long_examples.py writes both):

trace - rows without "stop_positions": the earlier recipe, cross-entropy on the completion (own thinking cut at
    the settled point, </think>, a right answer). The same loss as stopping/train_lora.py.

decision - rows with "stop_positions": the completion is the base model's whole rollout, and only the decision to
    end the thinking is trained. At a stop position the loss is -log p(</think>). At every other position it is
    the KL divergence from the base model's own next-token distribution (the same weights with the adapter
    switched off, no gradient), so the wording of the thinking, the answer and the decision to go on where the
    answer is not settled stay the base model's. Per sequence: mean KL over ordinary positions
    + stop_weight * mean stop loss + hold_weight * mean KL over the hold positions (boundaries where it must go on).
    After Settle (arXiv 2609.38997), which trains the end-of-thinking probability at boundaries with a KL term
    to the base elsewhere; the details here (which boundaries, the weighting) are this project's.

Sequences longer than --max-length: a trace row is dropped; a decision row is cut to its first --max-length
tokens (every position of a prefix is still a valid training position). One sequence per forward pass, gradient
checkpointing, logits in chunks. The adapter is saved every --save-every optimizer steps and a run started again
with the same arguments continues from the last save (the optimizer's moments start again: noted in the log).
Adapter modules as in every earlier study (attention and MLP projections), loadable by vLLM.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import chat  # noqa: E402

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.8-27B")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LOSS_CHUNK = 512


def load_rows(path, tokenizer, max_length):
    """Rows -> dicts(ids, prompt_length, stops or None). stops are indices into the completion."""
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    examples, dropped, cut_short = [], 0, 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt = chat.prompt_ids(tokenizer, row["prompt"])
            completion = list(row["completion_ids"])
            stops = row.get("stop_positions")
            if stops is None:
                if completion[-1] != end:
                    completion.append(end)              # the model must also learn to end the reply
                if len(prompt) + len(completion) > max_length:
                    dropped += 1
                    continue
            else:
                if row.get("finished") and completion[-1] != end:
                    completion.append(end)
                if len(prompt) + len(completion) > max_length:
                    completion = completion[:max_length - len(prompt)]
                    cut_short += 1
                stops = [s for s in stops if s < len(completion)]
            holds = [h for h in row.get("hold_positions", []) if h < len(completion)]
            examples.append({"ids": prompt + completion, "prompt_length": len(prompt), "stops": stops, "holds": holds})
    return examples, dropped, cut_short


def hidden_states(model, ids):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base, base.model(input_ids=ids, use_cache=False).last_hidden_state[0]


def head_backward(base, states, part, scale):
    """Loss and backward pass through the output layer in chunks of positions. The logits of a long sequence do
    not fit next to each other (a megabyte per position), so every chunk's graph is freed at once: the gradient
    is collected at the hidden states and sent back through the transformer in one go. part(logits, start) ->
    the chunk's summed loss, already divided by its normaliser. Returns the loss as a number."""
    leaf = states.detach().requires_grad_(True)
    total = 0.0
    for start in range(0, leaf.shape[0], LOSS_CHUNK):
        loss = part(base.lm_head(leaf[start:start + LOSS_CHUNK]).float(), start) * scale
        loss.backward()
        total += loss.item()
    states.backward(leaf.grad)
    return total / scale


def trace_loss(model, ids, prompt_length, scale):
    """Mean cross-entropy over the completion (position t predicts token t+1)."""
    base, hidden = hidden_states(model, ids)
    states, targets = hidden[prompt_length - 1:-1], ids[0, prompt_length:]
    part = lambda logits, start: F.cross_entropy(logits, targets[start:start + LOSS_CHUNK], reduction="sum") / targets.numel()  # noqa: E731
    return head_backward(base, states, part, scale), {}


def decision_loss(model, ids, prompt_length, stops, holds, close_id, weights, scale):
    """Three terms, each a mean over its own positions (completion index j is hidden position prompt_length-1+j):
    -log p(</think>) at the stop positions; KL(base || model) at the hold positions (boundaries where the model
    must go on); KL(base || model) at all other positions. The boundaries get their own terms because they are a
    handful among thousands of positions: inside one mean over the sequence the "go on" signal at a boundary
    weighs nothing against the "stop" term, and the model learns to stop at the first boundary it meets
    (seen with Qwen3-0.6B on 2026-11-23: thinking fell from 2 236 to 109 tokens)."""
    with torch.no_grad(), model.disable_adapter():
        base, reference = hidden_states(model, ids)
        reference = reference[prompt_length - 1:-1].detach()
    base, hidden = hidden_states(model, ids)
    states = hidden[prompt_length - 1:-1]
    kind = torch.zeros(states.shape[0], dtype=torch.long, device=states.device)      # 0 other, 1 stop, 2 hold
    if holds:
        kind[torch.tensor(holds, device=states.device)] = 2
    if stops:
        kind[torch.tensor(stops, device=states.device)] = 1
    others, n_hold = max(1, int((kind == 0).sum())), int((kind == 2).sum())
    stop_weight, hold_weight = weights
    seen = {"kl": 0.0, "stop": 0.0, "p_stop": 0.0, "hold": 0.0, "p_hold": 0.0}

    def part(logits, start):
        k = kind[start:start + LOSS_CHUNK]
        log_p = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            log_q = F.log_softmax(base.lm_head(reference[start:start + LOSS_CHUNK]).float(), dim=-1)
        kl = (log_q.exp() * (log_q - log_p)).sum(-1)
        seen["kl"] += kl[k == 0].sum().item()
        loss = kl[k == 0].sum() / others
        if (k == 1).any():
            stop = -log_p[k == 1, close_id].sum()
            seen["stop"] += stop.item()
            seen["p_stop"] += log_p[k == 1, close_id].exp().sum().item()
            loss = loss + stop_weight * stop / len(stops)
        if (k == 2).any():
            seen["hold"] += kl[k == 2].sum().item()
            seen["p_hold"] += log_p[k == 2, close_id].exp().sum().item()
            loss = loss + hold_weight * kl[k == 2].sum() / n_hold
        return loss
    loss = head_backward(base, states, part, scale)
    count = max(1, len(stops))
    return loss, {"kl": seen["kl"] / others, "stop": seen["stop"] / count, "p_stop": seen["p_stop"] / count,
                  "stops": len(stops), "hold": seen["hold"] / max(1, n_hold), "p_hold": seen["p_hold"] / max(1, n_hold),
                  "holds": n_hold}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16, help="Sequences per optimizer step")
    parser.add_argument("--max-length", type=int, default=40000)
    parser.add_argument("--stop-weight", type=float, default=1.0)
    parser.add_argument("--hold-weight", type=float, default=4.0)
    parser.add_argument("--save-every", type=int, default=20, help="Optimizer steps between saves")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    close_id = chat.think_ids(tokenizer)[1]
    examples, dropped, cut_short = load_rows(arguments.data, tokenizer, arguments.max_length)
    objective = "decision" if examples and examples[0]["stops"] is not None else "trace"
    tokens = sum(len(e["ids"]) - e["prompt_length"] for e in examples)
    print(f"objective {objective}: {len(examples)} examples ({dropped} over {arguments.max_length} tokens dropped, "
          f"{cut_short} cut to {arguments.max_length}), {tokens / 1e6:.2f} M completion tokens"
          + (f", {sum(len(e['stops']) for e in examples)} stop positions in {sum(bool(e['stops']) for e in examples)} examples"
             if objective == "decision" else ""), flush=True)

    steps = math.ceil(len(examples) * arguments.epochs / arguments.batch)
    state_path, first = arguments.out / "training_state.json", 0
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if state_path.exists() and (arguments.out / "adapter_config.json").exists():
        first = json.loads(state_path.read_text())["step"]
        model = PeftModel.from_pretrained(model, str(arguments.out), is_trainable=True)
        print(f"continuing from step {first} of {steps} (saved adapter; optimizer moments start again)", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=arguments.rank, lora_alpha=arguments.alpha, lora_dropout=0.05,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=arguments.lr, weight_decay=0.0)
    warmup = max(1, steps // 20)
    factor = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))  # noqa: E731
    rng = random.Random(arguments.seed)
    order = []
    while len(order) < steps * arguments.batch:
        epoch = list(range(len(examples)))
        rng.shuffle(epoch)
        order += epoch

    def save(step):
        arguments.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(arguments.out, safe_serialization=True)
        state_path.write_text(json.dumps({"step": step, "steps": steps}))

    model.train()
    started, seen = time.monotonic(), 0
    for step in range(first, steps):
        for group in optimizer.param_groups:
            group["lr"] = arguments.lr * factor(step)
        running, parts = 0.0, {}
        for index in order[step * arguments.batch:(step + 1) * arguments.batch]:
            example = examples[index]
            ids = torch.tensor([example["ids"]], device="cuda")
            if objective == "trace":
                loss, detail = trace_loss(model, ids, example["prompt_length"], 1 / arguments.batch)
            else:
                loss, detail = decision_loss(model, ids, example["prompt_length"], example["stops"], example["holds"],
                                             close_id, (arguments.stop_weight, arguments.hold_weight), 1 / arguments.batch)
            running += loss / arguments.batch
            seen += ids.shape[1]
            for key, value in detail.items():
                parts.setdefault(key, []).append(value)
            del ids
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % arguments.save_every == 0 and step + 1 < steps:
            save(step + 1)
        if step % 5 == 0 or step == steps - 1:
            minutes = (time.monotonic() - started) / 60
            extra = ""
            if parts:
                with_stop = [k for k, n in enumerate(parts["stops"]) if n]
                mean = lambda key, rows: sum(parts[key][k] for k in rows) / max(1, len(rows))  # noqa: E731
                with_hold = [k for k, n in enumerate(parts["holds"]) if n]
                extra = (f"  kl {mean('kl', range(len(parts['kl']))):.4f}  stop {mean('stop', with_stop):.3f}  "
                         f"p(stop) {mean('p_stop', with_stop):.3f}  hold kl {mean('hold', with_hold):.3f}  "
                         f"p(stop at hold) {mean('p_hold', with_hold):.3f}")
            print(f"step {step + 1}/{steps}  loss {running:.4f}{extra}  lr {arguments.lr * factor(step):.2e}  "
                  f"{minutes:.0f} min, ~{minutes / (step + 1 - first) * (steps - step - 1):.0f} min left, "
                  f"{seen / max(1e-9, minutes * 60):.0f} tokens/s", flush=True)

    save(steps)
    (arguments.out / "training_args.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(arguments).items()} | {"examples": len(examples), "objective": objective,
                                                           "base_model": MODEL}, indent=2))
    print(f"saved {arguments.out}")


if __name__ == "__main__":
    main()
