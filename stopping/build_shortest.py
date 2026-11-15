"""Reference baseline data: for each question, the shortest of the model's own correct rollouts.

    python stopping/build_shortest.py runs/rollouts_gsm8k --out runs/sft_shortest.jsonl

This is the simplest way to make a model shorter with its own words (self-training
on best-of-N, as in "Self-Training Elicits Concise Reasoning"), and the bar the
stop-point method has to clear. A rollout qualifies only if it finished on its own
and passes the strict grader, so the model also learns to end on a clean
"Final answer:" line, the format the evaluation reads.

Each output row is the prompt (chat messages) and the rollout's own token ids, so
training sees exactly what the model generated.

Hard MATH questions (source "math", Amendment 2b in the research log): Qwen3-1.7B solves them but closes
with a display `\\boxed{N}` instead of the requested "Final answer: N" line, so almost none pass the strict
grader. For these, when no rollout passes strictly, the shortest finished rollout that knows the answer is
used with only its closing `\\boxed{N}` line replaced by "Final answer: N" (thinking untouched, the rest of
the answer kept), and the result must pass the strict grader.

From the 4B study on this covers every math question (GSM8K too), and a decorated closing line
("Final answer: $42$", "**Final answer:** 42.") is rewritten the same way: the strict grade of a math
answer is now the bare line "Final answer: N" (stopping/answers.py, is_plain_correct), so that every
math training row ends in exactly the form the prompt asks for. Regrade old rollout files first
(stopping/regrade_rollouts.py) so that their stored `correct` is this grade.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.answers import clean_answer, same_number  # noqa: E402
from stopping.grading import grade  # noqa: E402

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")
# A closing line that holds only the boxed value, optionally in $ or $$ (also on lines of their own) and a period.
CLOSING_BOXED = re.compile(r"(?:^\$\$[ \t]*\n)?^[ \t]*\$*[ \t]*\\boxed\{([^{}]*)\}[ \t]*\$*[ \t]*\.?[ \t]*"
                           r"(?:\n\$\$[ \t]*)?\s*\Z", re.M)
FINAL_HEADING = re.compile(r"\n+#+[ \t]*\**Final Answer\**:?[ \t]*\Z", re.I)
# A closing "Final answer:" line whose value is decorated (emphasis marks, $, a box, a trailing period).
CLOSING_FINAL = re.compile(r"^[ \t]*\**Final answer:\**[ \t]*([^\n]+?)[ \t]*\Z", re.M)


def normalized_answer(answer, gold):
    """Answer text ending in a boxed line -> the same text ending in 'Final answer: N', or None."""
    answer = answer.rstrip()
    match = CLOSING_BOXED.search(answer) or CLOSING_FINAL.search(answer)
    if not match:
        return None
    value = clean_answer(match.group(1).strip().replace("**", ""))
    if not same_number(value, gold):
        return None
    head = FINAL_HEADING.sub("", answer[:match.start()].rstrip())
    return f"{head}\n\nFinal answer: {gold}".lstrip()


def normalized_ids(row, tokenizer, close_id):
    ids = row["token_ids"]
    if close_id not in ids:
        return None
    cut = ids.index(close_id) + 1
    ending = [ids[-1]] if ids[-1] in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")) else []
    answer = tokenizer.decode(ids[cut:len(ids) - len(ending)], skip_special_tokens=True)
    fixed = normalized_answer(answer, row["answer"])
    if fixed is None:
        return None
    new = ids[:cut] + tokenizer.encode(fixed, add_special_tokens=False) + ending
    text = tokenizer.decode(new, skip_special_tokens=True)
    return new if grade([(row, text)])[0][0] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path, help="A directory written by generate_rollouts.py")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    by_question = defaultdict(list)
    for line in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines():
        row = json.loads(line)
        by_question[row["question_id"]].append(row)

    tokenizer = None
    chosen, mean_lengths, normalized = [], [], 0
    for rows in by_question.values():
        good = [r for r in rows if r["correct"] and r["finished"]]
        if good:
            best = min(good, key=lambda r: r["tokens"])
            ids = best["token_ids"]
        elif rows[0]["task"] == "math":
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(MODEL)
                close_id = tokenizer.convert_tokens_to_ids("</think>")
            fixed = [(r, normalized_ids(r, tokenizer, close_id))
                     for r in sorted(rows, key=lambda r: r["tokens"]) if r["knows"] and r["finished"]]
            fixed = [(r, ids) for r, ids in fixed if ids is not None]
            if not fixed:
                continue
            best, ids = fixed[0]
            normalized += 1
        else:
            continue
        chosen.append({"question_id": best["question_id"], "source": best["source"],
                       "prompt": best["prompt"], "completion_ids": ids, "tokens": len(ids)})
        mean_lengths.append(statistics.mean(r["tokens"] for r in rows))

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text("".join(json.dumps(r) + "\n" for r in chosen))
    picked = [r["tokens"] for r in chosen]
    print(f"{len(chosen)} of {len(by_question)} questions have a correct, finished rollout "
          f"({normalized} of them with the closing line rewritten to 'Final answer: N')")
    print(f"chosen: mean {statistics.mean(picked):.0f} tokens, median {statistics.median(picked):.0f}; "
          f"all rollouts of those questions: mean {statistics.mean(mean_lengths):.0f}")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
