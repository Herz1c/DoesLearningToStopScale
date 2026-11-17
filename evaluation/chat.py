"""The few facts about a model's chat template and tokenizer that the rest of the code depends on.

Qwen3 leaves the opening <think> to the model: a reply is "<think> ... </think> answer". The Qwen3.5
family (Qwen3.5, 3.6, 3.8) writes "<think>\\n" into the prompt, so a reply starts inside the thinking
and is "... </think> answer". Everything that splits, counts or rebuilds a reply asks here instead of
assuming one of the two. Also here because transformers 5 changed what
apply_chat_template(tokenize=True) returns: prompts are rendered as text and tokenised explicitly.
"""

from functools import lru_cache

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def prompt_text(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def prompt_ids(tokenizer, messages):
    """The generation prompt as a plain list of token ids, the same in transformers 4 and 5."""
    return list(tokenizer(prompt_text(tokenizer, messages), add_special_tokens=False)["input_ids"])


def think_ids(tokenizer):
    """(id of <think>, id of </think>); both are single tokens in every Qwen3-or-later tokenizer."""
    ids = [tokenizer.encode(tag, add_special_tokens=False) for tag in (THINK_OPEN, THINK_CLOSE)]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"<think> or </think> is not a single token for {tokenizer.name_or_path}: {ids}")
    return ids[0][0], ids[1][0]


def blank_line_ids(tokenizer):
    """Token ids of the blank line the model writes after </think>."""
    return list(tokenizer.encode("\n\n", add_special_tokens=False))


@lru_cache(maxsize=8)
def _opens(name, rendered):
    return rendered.rstrip().endswith(THINK_OPEN)


def prompt_opens_thinking(tokenizer):
    """True when the template itself writes <think> at the end of the prompt (Qwen3.5 family)."""
    return _opens(tokenizer.name_or_path, prompt_text(tokenizer, [{"role": "user", "content": "Hello"}]))


def whole_reply(tokenizer, text):
    """Reply text in the one shape the graders know, "<think> ... </think> answer", whichever way the
    template splits it between prompt and reply."""
    if prompt_opens_thinking(tokenizer) and not text.lstrip().startswith(THINK_OPEN):
        return THINK_OPEN + "\n" + text
    return text


def engine_options(model):
    """Extra vLLM engine arguments a model family needs. The Qwen3.5 family is image-and-text: it is
    served text-only, without its vision tower's inputs. Qwen3 needs nothing (and gets nothing, so
    the earlier studies run exactly as they did)."""
    if model.startswith("Qwen/Qwen3-"):
        return {}
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model)
    return {"limit_mm_per_prompt": {"image": 0, "video": 0}} if hasattr(config, "vision_config") else {}


def leave(status=0):
    """Ends a script that ran a vLLM engine: stops the engine's processes, then exits without the
    interpreter's teardown (which has hung after long runs). A plain os._exit left the engine process
    alive on the 27B machine, still holding the card's memory when the next job started (out of memory
    in the calibration of 2026-11-17)."""
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        import psutil
        children = psutil.Process().children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(children, timeout=60)
    except Exception as error:      # never let the clean-up turn a finished run into a failed one
        print(f"note: engine processes not stopped ({type(error).__name__}: {error})", file=sys.stderr)
    os._exit(status)
