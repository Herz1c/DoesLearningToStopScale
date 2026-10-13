"""Harder math training questions: MATH (Hendrycks et al.) train split, levels 3-5, integer answers.

    python stopping/math_tasks.py --train 1500 --val 100     # writes data/math_hard_train.jsonl, data/math_hard_val.jsonl

GSM8K is nearly saturated for Qwen3-1.7B (80% of questions answered right in 4 of 4 tries), so the
math pool is mixed with these to give the same difficulty profile as GSM8K gave Qwen3-0.6B
(Amendment 2 in the research log). Source: EleutherAI/hendrycks_math (MIT licence), train split
only; the MATH test split, from which MATH-500 is drawn for the held-out check, is never used.
The answer is the last \\boxed{} of the reference solution; only whole-number answers are kept, so
grading is the same exact comparison as for GSM8K (stopping/answers.py). Rows have the format of
data/train.jsonl with source "math" and the problem's level and type.
"""

import argparse
import json
from pathlib import Path
import random
import re

from datasets import load_dataset

INSTRUCTION = "Put the final answer on the last line, like: Final answer: 42"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
            "prealgebra", "precalculus")
LEVELS = {"Level 3", "Level 4", "Level 5"}


def last_boxed(solution):
    """Content of the last \\boxed{...}, braces matched."""
    start = solution.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len("\\boxed")
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i + 1:j]
    return None


def integer(answer):
    if answer is None:
        return None
    answer = answer.strip().replace(",", "").replace("\\!", "").strip("$ ")
    return answer if re.fullmatch(r"-?\d+", answer) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--val", type=int, default=100)
    arguments = parser.parse_args()
    rows = []
    for subject in SUBJECTS:
        for r in load_dataset("EleutherAI/hendrycks_math", subject, split="train"):
            answer = integer(last_boxed(r["solution"]))
            if r["level"] in LEVELS and answer is not None:
                rows.append({"prompt": [{"role": "user", "content": f"{r['problem']}\n\n{INSTRUCTION}"}],
                             "answer": answer, "task": "math", "source": "math",
                             "level": r["level"], "type": r["type"]})
    print(f"{len(rows)} MATH train problems, levels 3-5, integer answers", flush=True)
    random.Random(0).shuffle(rows)
    val, train = rows[:arguments.val], rows[arguments.val:arguments.val + arguments.train]
    Path("data").mkdir(exist_ok=True)
    for name, part in (("math_hard_train", train), ("math_hard_val", val)):
        Path(f"data/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        print(f"data/{name}.jsonl: {len(part)} rows")


if __name__ == "__main__":
    main()
