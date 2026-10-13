"""Scoring one reply. Each grader returns (correct, parse_error); parse_error is None when the reply
was read and simply wrong, and names the reason when it could not be read.

    SVAMP   our rule: the last line after </think> is "Final answer: <number>", compared as a decimal
    BFCL    the reply parsed as a Python list of calls (never executed), judged by BFCL's AST checker
    IFEval  Google's strict verifiers (evaluation_lib.test_instruction_following_strict)
    MBPP+   EvalPlus `sanitize` extracts the program; it is then run by evaluation/execution.py

Only the text after </think> is graded, except that SVAMP checks the thinking block is well formed.
A reply cut off by the token cap is wrong on SVAMP, BFCL and MBPP+. IFEval grades whatever answer
text there is, capped or not (as the reference harness does; a capped IFEval answer is usually a
loop that fails its constraints anyway).
"""

import ast
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import random
import re
import sys
import types

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


# ---------------------------------------------------------------- reasoning / answer split

def split_reply(token_ids, open_id, close_id, eos_ids):
    """Generated ids -> (answer ids, reasoning token count, error).

    A well-formed reply is "<think> ... </think> answer" (the chat template leaves <think> to the
    model). No tags at all: the whole reply is the answer. Anything else has no defined answer.
    """
    ids = [t for t in token_ids if t not in eos_ids]
    opens = [i for i, t in enumerate(ids) if t == open_id]
    closes = [i for i, t in enumerate(ids) if t == close_id]
    if not opens and not closes:
        return ids, 0, None
    if opens != [0] or len(closes) != 1:
        return [], len(ids), "unbalanced_thinking"
    return ids[closes[0] + 1:], closes[0] + 1, None


# ---------------------------------------------------------------- SVAMP

_ANSWER_LINE = re.compile(r"Final answer:[ \t]*([^\n]+)")
_NUMBER = re.compile(r"[+-]?(\d+|\d{1,3}(,\d{3})+)(\.\d+)?")


def svamp_answer(text):
    """Whole reply -> (the value on the closing 'Final answer:' line, None) or (None, reason)."""
    text = text.strip()
    if THINK_OPEN in text or THINK_CLOSE in text:
        if text.count(THINK_OPEN) != 1 or text.count(THINK_CLOSE) != 1 or not text.startswith(THINK_OPEN):
            return None, "malformed_thinking_block"
        text = text.split(THINK_CLOSE, 1)[1].strip()
    answers = [m for m in map(_ANSWER_LINE.match, text.split("\n")) if m]
    if not answers:
        return None, "missing_final_answer"
    if not _ANSWER_LINE.fullmatch(text.splitlines()[-1]):
        return None, "trailing_content"
    return answers[-1].group(1).strip(), None


def as_decimal(text):
    """'1,234.5' -> Decimal('1234.5'); units, words, fractions, LaTeX -> None."""
    text = text.strip()
    return Decimal(text.replace(",", "")) if _NUMBER.fullmatch(text) else None


def grade_svamp(text, gold, truncated):
    if truncated:
        return False, None
    value, error = svamp_answer(text)
    if error:
        return False, error
    expected = as_decimal(gold)
    if expected is None:
        raise ValueError(f"gold answer is not a number: {gold!r}")
    got = as_decimal(value)
    if got is None:
        return False, "invalid_numeric_answer"
    return got == expected, None


# ---------------------------------------------------------------- BFCL

_FENCE = "```"
_FENCED = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)


def parse_calls(reply):
    """'[f(a=1), g.h(b="x")]' -> [{'f': {'a': 1}}, {'g.h': {'b': 'x'}}]; raises ValueError/SyntaxError.

    Parsed with ast, never evaluated: only keyword arguments with literal values are accepted.
    A reply wrapped in one ```python fence is unwrapped first.
    """
    reply = reply.strip()
    if reply.startswith(_FENCE):
        blocks = _FENCED.findall(reply)
        if reply.count(_FENCE) != 2 or len(blocks) != 1:
            raise ValueError("ambiguous_code_fences")
        reply = blocks[0]
    tree = ast.parse(reply, mode="eval").body
    calls = []
    for node in tree.elts if isinstance(tree, ast.List) else [tree]:
        if not isinstance(node, ast.Call) or node.args:
            raise ValueError("expected calls with keyword arguments only")
        arguments = {}
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in arguments:
                raise ValueError("**kwargs or a repeated argument")
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        calls.append({_dotted(node.func): arguments})
    return calls


def _dotted(node):
    names = []
    while isinstance(node, ast.Attribute):
        names.insert(0, node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise ValueError("function name is not a (dotted) name")
    return ".".join([node.id] + names)


@contextmanager
def _on_path(directory):
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        if str(directory) in sys.path:
            sys.path.remove(str(directory))


def load_bfcl_checker(bfcl_root):
    """BFCL's ast_checker, imported from the pinned files.

    Its module imports bfcl_eval.constants.model_config only to look up whether a provider rewrites
    dots in function names; that module pulls in every API client, so it is replaced by a stub
    entry for our local model, which keeps names as they are.
    """
    stub = types.ModuleType("bfcl_eval.constants.model_config")
    stub.MODEL_CONFIG_MAPPING = {"local": types.SimpleNamespace(underscore_to_dot=False)}
    sys.modules["bfcl_eval.constants.model_config"] = stub
    with _on_path(bfcl_root):
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    def check(row, calls):
        return bool(ast_checker(row["function"], calls, row["reference"], Language.PYTHON,
                                row["category"], "local")["valid"])
    return check


def grade_bfcl(row, response, check, truncated):
    """Irrelevance (BFCL's rule): right unless the reply is a parseable, non-empty call list."""
    if truncated:
        return False, "truncated"
    try:
        calls = parse_calls(response)
    except (ValueError, SyntaxError, TypeError, RecursionError) as error:
        if row["category"] == "irrelevance":
            return True, None
        return False, f"{type(error).__name__}: {error}"
    if row["category"] == "irrelevance":
        return calls == [], None
    return check(row, calls), None


# ---------------------------------------------------------------- IFEval

def load_ifeval(ifeval_package):
    """grade(row, response) with Google's strict check. ifeval_package = .../instruction_following_eval."""
    import langdetect
    langdetect.DetectorFactory.seed = 0      # language detection is randomised unless seeded
    with _on_path(Path(ifeval_package).parent):
        from instruction_following_eval import evaluation_lib

    def grade(row, response):
        example = evaluation_lib.InputExample(key=row["id"], instruction_id_list=row["instruction_id_list"],
                                              prompt=row["original_prompt"], kwargs=row["kwargs"])
        # Verifiers draw a random value for any argument the data leaves unset; pin the draw per question.
        random.seed(f"ifeval:{row['id']}")
        result = evaluation_lib.test_instruction_following_strict(example, {row["original_prompt"]: response})
        return bool(result.follow_all_instructions), None
    return grade


# ---------------------------------------------------------------- MBPP+

def load_sanitize(evalplus_root):
    with _on_path(evalplus_root):
        from evalplus.sanitize import sanitize
    return sanitize


def mbpp_program(response, entry_point, sanitize, truncated):
    """-> (program, error). EvalPlus keeps the code the entry point needs and drops the rest."""
    if truncated:
        return "", "truncated"
    program = sanitize(response, entry_point)
    return (program, None) if program.strip() else ("", "empty_code")
