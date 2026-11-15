"""Answer checking for the training questions (GSM8K math, multiple choice). Not the benchmarks:
those are graded by evaluation/graders.py.

    is_task_correct(completion, "42", "math")   strict: as the prompt asks, one "Final answer:" line,
                                                and it is the last line; \\boxed{}, $, ** and a
                                                trailing period around the value are tolerated
    knows(completion, "42", "math")             lenient: the last "Final answer:" value, else the
                                                last \\boxed{}, anywhere after </think>

The training prompt (data/train.jsonl) says "Put the final answer on the last line, like: Final
answer: 42". The strict check accepts more than the SVAMP grader does (\\boxed{42}, 84/2, $42),
so a strict pass here is not a guarantee of a SVAMP point; `knows` separates knowing the answer
from formatting it (analysis/forced_stop_sweep.py). Same rules as the Qwen3-0.6B study.
"""

from fractions import Fraction
import re

_LINE = re.compile(r"^Final answer:[ \t]*([^\n]+?)[ \t]*$", re.M)
_ANY_FINAL = re.compile(r"Final answer:[ \t]*([^\n]+)")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_WRAPPED = re.compile(r"\\(?:boxed|text)\{(.*)\}")
_TAGS = re.compile(r"</?answer>")
_DEGREES = re.compile(r"(?:°|\^\\circ|\^\{\\circ\})$")
_LABEL = re.compile(r"\(?([A-Za-z0-9])\b\)?")


def final_answer(response):
    """Answer text (after </think>) -> the value of its single, closing 'Final answer:' line, or None."""
    response = response.strip()
    if "<think>" in response or "</think>" in response:
        return None
    values = _LINE.findall(response)
    if len(values) != 1 or not response.splitlines()[-1].startswith("Final answer:"):
        return None
    return values[0].strip()


def clean_answer(value):
    """Peel \\boxed{}/\\text{}, <answer> tags, *, $, <, > and a trailing period until nothing changes."""
    value = _TAGS.sub("", value)
    while True:
        wrapped = _WRAPPED.fullmatch(value)
        peeled = (wrapped.group(1) if wrapped else value).strip(" *$<>").rstrip(".")
        if peeled == value:
            return value
        value = peeled


def as_fraction(text):
    text = _DEGREES.sub("", text.replace(",", "").replace("$", "").strip())
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def same_number(predicted, gold):
    a, b = as_fraction(predicted), as_fraction(gold)
    if a is None or b is None:
        return " ".join(predicted.casefold().split()) == " ".join(gold.casefold().split())
    return a == b


def same_choice(predicted, gold):
    label = _LABEL.match(predicted)
    return label is not None and label.group(1).upper() == gold.strip().upper()


def matches(value, gold, task):
    if task == "math":
        return same_number(value, gold)
    if task == "multiple_choice":
        return same_choice(value, gold)
    raise ValueError(f"unknown task {task!r}")


def is_task_correct(completion, gold, task):
    value = final_answer(completion.split("</think>")[-1].replace("**", ""))
    return value is not None and matches(clean_answer(value), gold, task)


_PLAIN_NUMBER = re.compile(r"[+-]?(\d+|\d{1,3}(,\d{3})+)(\.\d+)?")


def is_plain_correct(completion, gold):
    """Strict in the narrow sense: the thinking is closed exactly once, the reply's last line is
    'Final answer: N' and N is a bare number (no \\boxed{}, $, unit, expression or trailing period).

    This is what "Put the final answer on the last line, like: Final answer: 42" literally asks for.
    `is_task_correct` forgives decorations; the 1.7B model then picked up decorated closing lines
    ("Final answer: 5 + 3 = 8", a second </think>), which a grader that wants the number alone
    rejects. From the 4B study on, the strict grade of a math answer is this one (stopping/grading.py)."""
    if completion.count("</think>") > 1:
        return False
    value = final_answer(completion.split("</think>")[-1])
    return value is not None and _PLAIN_NUMBER.fullmatch(value) is not None and same_number(value, gold)


def knows(completion, gold, task):
    response = completion.split("</think>")[-1].replace("**", "")
    finals, boxed = _ANY_FINAL.findall(response), _BOXED.findall(response)
    value = finals[-1] if finals else boxed[-1] if boxed else None
    return value is not None and matches(clean_answer(value.strip()), gold, task)
