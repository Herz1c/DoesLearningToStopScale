"""Runs model-written programs against LiveCodeBench tests, one stream per process.

Reads one JSON object per line on stdin, {"key", "code", "tests": [{"input", "output", "testtype"}], "func_name"},
and writes one verdict per line: {"key", "passed": bool, "error": str or null}.

Two kinds of test, as in the benchmark: "stdin" (the program reads the input and prints the answer; outputs are
compared line by line, numbers with a small tolerance) and "functional" (the method `func_name` of the class
`Solution` is called with the JSON-decoded arguments, one per input line, and its result compared with the
JSON-decoded output). Every test runs in its own Python process with a time limit; the first failing test ends
the program's run. Executes untrusted code: the caller provides the sandbox (stopping/code_tasks.py).
"""

import json
import os
import subprocess
import sys
import tempfile

TIME_LIMIT = 6          # seconds per test, the benchmark's default
PRELUDE = ("import sys, os, re, math, heapq, bisect, itertools, functools, collections, string, random, operator\n"
           "from typing import *\nfrom collections import *\nfrom itertools import *\nfrom functools import *\n"
           "from heapq import *\nfrom bisect import *\nfrom math import *\nfrom builtins import *\nsys.setrecursionlimit(1000000)\n")
CALLER = """
import json as _json, sys as _sys
_args = [_json.loads(line) for line in _sys.stdin.read().split("\\n") if line.strip()]
_result = Solution().{name}(*_args)
if isinstance(_result, tuple):
    _result = list(_result)
print("\\n__RESULT__" + _json.dumps(_result))
"""


def same_output(got, expected):
    got_lines = [line.strip() for line in got.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    if got_lines == expected_lines:
        return True
    if len(got_lines) != len(expected_lines):
        return False
    for a, b in zip(got_lines, expected_lines):
        if a == b:
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if x != y:
                try:
                    if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                        return False
                except ValueError:
                    return False
    return True


def run(path, text):
    try:
        done = subprocess.run([sys.executable, path], input=text, text=True, capture_output=True, timeout=TIME_LIMIT)
    except subprocess.TimeoutExpired:
        return None, "time limit"
    if done.returncode != 0:
        return None, "runtime error: " + done.stderr.strip().splitlines()[-1][:160] if done.stderr.strip() else "runtime error"
    return done.stdout, None


def judge(item, directory):
    functional = bool(item["tests"]) and item["tests"][0].get("testtype") == "functional"
    source = PRELUDE + item["code"] + (CALLER.format(name=item["func_name"]) if functional else "")
    path = os.path.join(directory, "solution.py")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(source)
    for index, test in enumerate(item["tests"]):
        out, error = run(path, test["input"])
        if error:
            return False, f"test {index}: {error}"
        if functional:
            marker = out.rfind("__RESULT__")
            try:
                if marker < 0 or json.loads(out[marker + len("__RESULT__"):]) != json.loads(test["output"]):
                    return False, f"test {index}: wrong answer"
            except json.JSONDecodeError:
                return False, f"test {index}: wrong answer"
        elif not same_output(out, test["output"]):
            return False, f"test {index}: wrong answer"
    return True, None


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        item = json.loads(line)
        with tempfile.TemporaryDirectory() as directory:
            try:
                passed, error = judge(item, directory)
            except Exception as failure:     # noqa: BLE001 -- any failure of the judge is a verdict
                passed, error = False, f"judge: {type(failure).__name__}: {str(failure)[:160]}"
        print(json.dumps({"key": item["key"], "passed": passed, "error": error}), flush=True)


if __name__ == "__main__":
    main()
