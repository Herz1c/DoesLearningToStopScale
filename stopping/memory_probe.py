"""How long a sequence can be trained on one card, and how fast.

    python stopping/memory_probe.py                 # 8k ... 65k tokens, LoRA as in stopping/train_lora.py

Loads the base model with the LoRA of the recipe (rank 16, gradient checkpointing) and, for growing sequence
lengths, runs one forward pass without gradients and one forward-backward pass on random tokens, reporting the
time of each and the peak memory. Stops at the first length that does not fit. Nothing is saved. The numbers
decide the longest training example of the long-reasoning track and its cost.
"""

import argparse
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.train_lora import MODEL, TARGETS, chunked_loss  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", default="8192,16384,32768,49152,65536")
    arguments = parser.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    model.train()
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"model {MODEL} loaded: {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB of {total:.0f} GiB", flush=True)
    print(f"{'tokens':>7} {'forward s':>10} {'forward+backward s':>19} {'peak GiB':>9} {'tokens/s trained':>17}", flush=True)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():       # warm-up, so that the first timed pass does not pay for start-up work
        chunked_loss(model, torch.randint(1000, 50000, (1, 512), generator=generator).to("cuda"), 64).item()
    for length in (int(x) for x in arguments.lengths.split(",")):
        ids = torch.randint(1000, 50000, (1, length), generator=generator).to("cuda")
        try:
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.no_grad():
                chunked_loss(model, ids, 64).item()
            forward = time.monotonic() - started
            started = time.monotonic()
            chunked_loss(model, ids, 64).backward()
            torch.cuda.synchronize()
            both = time.monotonic() - started
            model.zero_grad(set_to_none=True)
            print(f"{length:7} {forward:10.1f} {both:19.1f} {torch.cuda.max_memory_allocated() / 2 ** 30:9.1f} "
                  f"{length / both:17.0f}", flush=True)
        except torch.OutOfMemoryError:
            print(f"{length:7} does not fit (out of memory)", flush=True)
            break
        finally:
            del ids
            torch.cuda.empty_cache()
    print("memory probe end", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
