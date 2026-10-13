"""The evaluation protocol in one place: what the model is asked, how it samples, how far it may go.

Everything that changes what the model generates lives here, so two runs are comparable exactly when
they agree on this file. The protocol is the one of the ThinkingCap Test (BottleCapAI), which the
Qwen3-0.6B study was scored on; it is re-implemented in this package and checked against that
harness on the 0.6B baseline (PROGRESS.md, "Own evaluation module").

    from evaluation.protocol import SAMPLING, SAMPLES, SEED, token_cap
"""

SAMPLES = 8                       # generations per question
SEED = 9001                       # job seed = SEED + 1000 * seed_index + sample
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# Generation caps. A capped reply is wrong. MBPP+ writes whole programs and gets twice the room.
DEFAULT_CAP = 4096
CAPS = {"mbpp": 8192}

# vLLM engine settings that change which tokens come out at a fixed seed (batch composition and
# memory layout), so they are part of the protocol too: max_model_len = largest cap + PROMPT_ROOM,
# requests are submitted BATCH at a time in question order, prefix caching off.
PROMPT_ROOM = 4096
BATCH = 256
GPU_MEMORY = 0.85

# Base models are pinned to a Hub revision so a baseline can be regenerated later.
REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}

# What each benchmark's question says around the benchmark's own text. Changing a word changes
# every generation, so these are frozen; the question file's hash records them.
MBPP_INSTRUCTION = ("Write a complete Python solution to the following task. "
                    "Return the solution in one Python code block.")
MBPP_NAME = ("The tests call `{name}` and nothing else, so any other function name fails. "
             "Define `{name}`, spelled exactly like that.")
SVAMP_SUFFIX = ("\n\nThe last line of your reply must start with 'Final answer:' "
                "followed by the number alone.")
# BFCL: the leaderboard's default prompt-mode system prompt (format string below, templates from
# bfcl_eval/constants/default_prompts.py), with the question's functions as given in the data.
BFCL_PROMPT_FORMAT = "ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic"
BFCL_CATEGORIES = ("simple_python", "multiple", "irrelevance")
# IFEval: the prompt as it is.

# Benchmark -> datasets it reports (BFCL's three categories count as one benchmark in averages).
BENCHMARKS = {
    "mbpp": ("mbpp",),
    "bfcl": tuple(f"bfcl_{c}" for c in BFCL_CATEGORIES),
    "svamp": ("svamp",),
    "ifeval": ("ifeval",),
}
QUESTION_ORDER_SEED = 42          # shuffles each benchmark once; seed_index = position in the file


def token_cap(dataset):
    return CAPS.get(dataset, DEFAULT_CAP)


def benchmark_of(dataset):
    return "bfcl" if dataset.startswith("bfcl_") else dataset
