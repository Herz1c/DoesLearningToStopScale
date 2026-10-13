"""Runs MBPP+ programs with EvalPlus, one JSON line in, one verdict line out. Started by execution.py,
inside its sandbox; never import this from the evaluation process.

    stdin   {"key": 17, "task": "Mbpp/2", "program": "def similar_elements(...): ..."}
    stdout  {"key": 17, "base": "pass", "plus": "fail"}

The expected outputs of a task come from running its canonical solution once, the way EvalPlus's
own get_groundtruth does, and are reused for every program of that task.
"""

import contextlib
import io
import json
import os
import sys


def main():
    evalplus_root, mbpp_file = sys.argv[1], sys.argv[2]
    os.environ["MBPP_OVERRIDE_PATH"] = mbpp_file      # read by evalplus.data.mbpp at import
    sys.path.insert(0, evalplus_root)
    from evalplus.data import get_mbpp_plus
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import check_correctness
    from evalplus.gen.util import trusted_exec

    problems = get_mbpp_plus()
    expected = {}

    def expected_for(task):
        if task not in expected:
            problem, oracle = problems[task], {}
            for split in ("base", "plus"):
                oracle[split], oracle[split + "_time"] = trusted_exec(
                    problem["prompt"] + problem["canonical_solution"], problem[split + "_input"],
                    problem["entry_point"], record_time=True,
                    output_not_none=problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS)
            expected[task] = oracle
        return expected[task]

    for line in sys.stdin:
        if not line.strip():
            continue
        item = json.loads(line)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                oracle = expected_for(item["task"])
                # fast_check stops at the first failing input; it changes speed, not the verdict.
                result = check_correctness("mbpp", 0, problems[item["task"]], item["program"], oracle,
                                           fast_check=True)
            verdict = {"key": item["key"], "base": result["base"][0], "plus": result["plus"][0]}
        except Exception as error:      # one broken program must not take the batch down
            verdict = {"key": item["key"], "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
