"""Function-calling tasks for stop-point distillation, from Glaive function-calling v2.

    python stopping/tool_tasks.py build --train 1500 --val 100   # writes data/tool_train.jsonl, data/tool_val.jsonl
    # harder tiers for larger models (see "Harder tiers" below), from Glaive requests the first pool did not use:
    python stopping/tool_tasks.py variants --tier wide --out data/tool_wide.jsonl --exclude data/tool_train.jsonl data/tool_val.jsonl
    python stopping/tool_tasks.py variants --tier removed --out data/tool_removed.jsonl --exclude data/tool_train.jsonl data/tool_val.jsonl data/tool_wide.jsonl
    # pools built before 2026-11-14 carry a labelling error in their no-call tasks (see `relabel`):
    python stopping/tool_tasks.py relabel --pool data/tool_wide.jsonl --out data/tool_wide_fixed.jsonl

Source: glaiveai/glaive-function-calling-v2 (Apache-2.0). Each conversation offers one or
more functions as JSON schemas. The first user turn and the first assistant turn give a
task with a checkable answer:

- the assistant calls a function (`<functioncall> {"name": ..., "arguments": ...}`): the
  target is that call, with those arguments;
- the assistant makes no call (the request does not fit, or needs details the user has not
  given): the target is *no call*.

About 30% of kept tasks are no-call, close to the evaluation's BFCL mix (240 of 840).
BFCL itself is never used; analysis/overlap_check.py checks the selection against every
evaluation question.

The system prompt is our own wording and asks for the same answer syntax the evaluation
reads, a Python list of calls with keyword arguments: [func(a=1, b="x")].

Grading parses the reply with evaluation/graders.parse_calls (ast parsing and literal_eval
only; nothing is executed). A call is right when the names match and every argument matches
the reference (numbers by value, strings ignoring case and surrounding space). No-call
follows BFCL's rule: anything that is not a parseable call is correct. *knows* also accepts a
reply wrapped in a code fence; *strict* asks for the bare list the evaluation expects.

Harder tiers. Glaive offers one or two functions per request, which a larger model nearly always
gets right. Two tiers raise the difficulty with Glaive material only:

- wide (source "glaive_wide"): the same kind of task with 5 to 8 functions to choose from. The
  extra functions come from other Glaive conversations.
- removed (source "glaive_removed"): 6 to 10 functions, and the no-call tasks are call tasks whose
  needed function was taken out of the list, so a sensible request meets a list in which nothing
  fits. This is the case where the 1.7B model lost accuracy (it called something anyway).

An added function must not fit the request, or the label would be wrong. The filter is lexical: a
function is not added when its name or description shares a content word (compared on the first
five letters, generic verbs such as "get" or "calculate" aside) with the request or with any
function the conversation offered. It is a heuristic; the share of tasks the base model gets
right on each tier is measured before the tier is used.
"""

import argparse
import json
import keyword
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.graders import parse_calls  # noqa: E402

SYSTEM = ("You can use the functions listed below. Read the user's request. If one or more of the "
          "functions fulfil it, reply with only the call(s), as a Python list with keyword arguments: "
          "[function_name(parameter=value, ...)]. Write nothing else. If none of the functions fits the "
          "request, or a required value is missing, do not call anything and say why.\n\n"
          "Functions (JSON):\n{functions}")
CALL = re.compile(r"<functioncall>\s*(\{.*\})", re.DOTALL)
QUOTED_CALL = re.compile(r'"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*)\'\s*\}\s*$', re.DOTALL)


# ---------- grading ----------

def same_value(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return a == b


def matches(reply, target):
    """target: None for no call, else {"name": ..., "arguments": {...}}."""
    try:
        calls = parse_calls(reply)
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return target is None
    if target is None:
        return calls == []
    if len(calls) != 1:
        return False
    (name, arguments), = calls[0].items()
    return name == target["name"] and same_value(arguments, target["arguments"])


def grade_tool(row, text):
    """-> (strict, knows)."""
    if "</think>" not in text:
        return False, False
    response = text.split("</think>")[-1].strip()
    target = json.loads(row["answer"])
    knows = matches(response, target)
    strict = knows and not response.startswith("```")
    return strict, knows


# ---------- data ----------

def parse(sample, first_turn_only=False):
    """A Glaive sample -> (functions, user turn, target) or None.

    first_turn_only=True reads the labels as the pools built before 2026-11-14 did (see `relabel`)."""
    system = sample["system"].removeprefix("SYSTEM:").strip()
    start = system.find("{")
    if start < 0:
        return None
    functions, decoder, rest = [], json.JSONDecoder(), system[start:]
    while rest:
        try:
            obj, end = decoder.raw_decode(rest)
        except json.JSONDecodeError:
            return None
        functions.append(obj)
        rest = rest[end:].lstrip().lstrip(",").lstrip()
        if rest and not rest.startswith("{"):
            break
    if not functions or not all(isinstance(f, dict) and "name" in f for f in functions):
        return None
    turns = re.split(r"\n*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*", sample["chat"])
    pairs = [(turns[i], turns[i + 1].replace("<|endoftext|>", "").strip()) for i in range(1, len(turns) - 1, 2)]
    if len(pairs) < 2 or pairs[0][0] != "USER:" or pairs[1][0] != "ASSISTANT:":
        return None
    user = pairs[0][1]
    # The assistant often announces the call in one turn ("Sure, let me calculate that for you.") and
    # makes it in the next. Reading only the first turn labelled those tasks "no call" (found in the 4B
    # calibration, 2026-11-14: 7 825 of Glaive's 40 595 first turns without a call are of this kind).
    replies = []
    for role, text in pairs[1:]:
        if role != "ASSISTANT:":
            break
        replies.append(text)
    reply = replies[0] if first_turn_only else next((text for text in replies if CALL.search(text)), replies[0])
    call = CALL.search(reply)
    if call:
        # Glaive writes the arguments as a single-quoted JSON string inside the call object,
        # which is not valid JSON as a whole: read the name and the argument string apart.
        quoted = QUOTED_CALL.search(call.group(1))
        try:
            if quoted:
                name, arguments = quoted.group(1), json.loads(quoted.group(2))
            else:
                obj = json.loads(call.group(1))
                name, arguments = obj.get("name"), obj.get("arguments", {})
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(arguments, dict) or name not in {f["name"] for f in functions}:
            return None
        # A parameter called e.g. `from` cannot be written as a keyword argument in the answer
        # syntax ([f(from="USD")] does not parse), so no reply could be graded right.
        if not name.isidentifier() or any(not k.isidentifier() or keyword.iskeyword(k) for k in arguments):
            return None
        target = {"name": name, "arguments": arguments}
    else:
        target = None
    return functions, user, target


def build(arguments):
    from datasets import load_dataset
    rows = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    calls, none = [], []
    for sample in rows:
        parsed = parse(sample)
        if parsed is None:
            continue
        functions, user, target = parsed
        if not user or len(user) > 1500:
            continue
        row = {"prompt": [{"role": "system", "content": SYSTEM.format(functions=json.dumps(functions, indent=4))},
                          {"role": "user", "content": user}],
               "answer": json.dumps(target), "task": "tool", "source": "glaive"}
        (calls if target else none).append(row)
    print(f"{len(calls)} call tasks, {len(none)} no-call tasks", flush=True)
    rng = random.Random(0)
    rng.shuffle(calls)
    rng.shuffle(none)
    total = arguments.train + arguments.val
    n_none = round(total * 0.3)
    picked = calls[:total - n_none] + none[:n_none]
    rng.shuffle(picked)
    # Sanity: every reference answer, written in the answer syntax, must grade as right.
    for row in picked:
        target = json.loads(row["answer"])
        reply = "[]" if target is None else \
            f"[{target['name']}(" + ", ".join(f"{k}={v!r}" for k, v in target["arguments"].items()) + ")]"
        assert grade_tool(row, "</think>\n\n" + reply) == (True, True), (reply, row["answer"])
    val, train = picked[:arguments.val], picked[arguments.val:]
    Path("data").mkdir(exist_ok=True)
    for name, part in (("tool_train", train), ("tool_val", val)):
        Path(f"data/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        print(f"data/{name}.jsonl: {len(part)} rows ({sum(json.loads(r['answer']) is None for r in part)} no-call)")


# ---------- harder tiers ----------

GENERIC = {"get", "calculate", "search", "find", "generate", "create", "check", "convert", "send", "make", "add",
           "given", "based", "specific", "specified", "particular", "provided", "information", "details", "value",
           "number", "type", "name", "list", "data", "user", "function", "from", "with", "that", "this", "for",
           "the", "and", "what", "please", "need", "want", "help", "could", "would", "like", "some", "have", "know"}
SIZES = {"wide": (5, 8), "removed": (6, 10)}


def content_words(text):
    """Content words of a text or an identifier, cut to five letters so that inflections match."""
    found = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    return {w[:5] for w in found if len(w) >= 3 and w not in GENERIC}


def function_words(function):
    return content_words(function["name"]) | content_words(str(function.get("description", "")))


def answer_text(target):
    if target is None:
        return "[]"
    return f"[{target['name']}(" + ", ".join(f"{k}={v!r}" for k, v in target["arguments"].items()) + ")]"


def variants(arguments):
    from datasets import load_dataset
    used = {json.loads(line)["prompt"][-1]["content"] for path in arguments.exclude
            for line in Path(path).read_text().splitlines()}
    calls, none, catalogue = [], [], {}
    for sample in load_dataset("glaiveai/glaive-function-calling-v2", split="train"):
        parsed = parse(sample)
        if parsed is None:
            continue
        functions, user, target = parsed
        for f in functions:
            if f["name"].isidentifier():
                catalogue.setdefault(f["name"], f)
        if user and len(user) <= 1500 and user not in used:
            used.add(user)      # Glaive repeats requests; one task per request text
            (calls if target else none).append((functions, user, target))
    catalogue = [(f, function_words(f)) for f in catalogue.values()]
    print(f"{len(calls)} call tasks, {len(none)} no-call tasks not in the excluded pools; "
          f"{len(catalogue)} distinct functions to add", flush=True)
    rng = random.Random(arguments.seed)
    rng.shuffle(calls)
    rng.shuffle(none)
    n_none = round(arguments.count * 0.3)
    n_call = arguments.count - n_none
    tasks = [(t, "call") for t in calls[:n_call]]
    if arguments.tier == "removed":
        tasks += [(t, "removed") for t in calls[n_call:arguments.count]]
    else:
        tasks += [(t, "no_call") for t in none[:n_none]]
    low, high = SIZES[arguments.tier]
    out = []
    for (functions, user, target), kind in tasks:
        blocked = content_words(user) | {w for f in functions for w in function_words(f)}
        if kind == "removed":
            functions, target = [f for f in functions if f["name"] != target["name"]], None
        names = {f["name"] for f in functions}
        extra, want = [], rng.randint(low, high) - len(functions)
        for index in rng.sample(range(len(catalogue)), len(catalogue)):
            if len(extra) >= want:
                break
            f, words = catalogue[index]
            if f["name"] not in names and not words & blocked:
                extra.append(f)
                names.add(f["name"])
                blocked |= content_words(f["name"])     # the added functions differ from each other too
        offered = functions + extra
        rng.shuffle(offered)
        row = {"prompt": [{"role": "system", "content": SYSTEM.format(functions=json.dumps(offered, indent=4))},
                          {"role": "user", "content": user}],
               "answer": json.dumps(target), "task": "tool", "source": f"glaive_{arguments.tier}",
               "variant": kind, "functions": len(offered)}
        assert grade_tool(row, "</think>\n\n" + answer_text(target)) == (True, True), row["answer"]
        out.append(row)
    rng.shuffle(out)
    Path(arguments.out).parent.mkdir(exist_ok=True)
    Path(arguments.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    kinds = {k: sum(r["variant"] == k for r in out) for k in ("call", "no_call", "removed")}
    print(f"{arguments.out}: {len(out)} rows {kinds}, "
          f"{sum(r['functions'] for r in out) / len(out):.1f} functions per task")


def offered_names(row):
    return {f["name"] for f in json.loads(row["prompt"][0]["content"].split("Functions (JSON):\n", 1)[1])}


def relabel(arguments):
    """Corrects a pool built with the first-turn-only labels: a no-call task whose Glaive conversation
    makes the call one assistant turn later gets that call as its target. A request that Glaive has
    both ways (one conversation calls, another with the same functions does not) is dropped."""
    from datasets import load_dataset
    seen = {}
    for sample in load_dataset("glaiveai/glaive-function-calling-v2", split="train"):
        old = parse(sample, first_turn_only=True)
        if old is None or old[2] is not None:
            continue
        new = parse(sample)
        seen.setdefault(old[1], []).append((frozenset(f["name"] for f in old[0]), new[2] if new else None))
    rows = [json.loads(line) for line in Path(arguments.pool).read_text().splitlines()]
    out, count = [], {"kept": 0, "relabelled to a call": 0, "dropped, Glaive has it both ways": 0}
    for row in rows:
        if json.loads(row["answer"]) is None and row.get("variant") != "removed":
            names = offered_names(row)
            targets = [t for functions, t in seen.get(row["prompt"][-1]["content"], []) if functions <= names]
            called = [t for t in targets if t is not None]
            if called and len(called) < len(targets):
                count["dropped, Glaive has it both ways"] += 1
                continue
            if called:
                row = dict(row, answer=json.dumps(called[0]))
                if "variant" in row:
                    row["variant"] = "call"
                count["relabelled to a call"] += 1
                assert grade_tool(row, "</think>\n\n" + answer_text(called[0])) == (True, True), row["answer"]
            else:
                count["kept"] += 1
        out.append(row)
    Path(arguments.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{arguments.pool}: no-call tasks {count}; {arguments.out}: {len(out)} rows, "
          f"{sum(json.loads(r['answer']) is None for r in out)} no-call")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("relabel")
    r.add_argument("--pool", required=True)
    r.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train", type=int, default=1500)
    b.add_argument("--val", type=int, default=100)
    v = sub.add_parser("variants")
    v.add_argument("--tier", choices=sorted(SIZES), required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--count", type=int, default=1600)
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--exclude", nargs="*", default=[], help="Pool files whose requests are left out")
    arguments = parser.parse_args()
    {"build": build, "variants": variants, "relabel": relabel}[arguments.command](arguments)


if __name__ == "__main__":
    main()
