"""The supervised training set ("mix G"): one file from the per-domain example files.

    python stopping/build_mix.py --runs runs/4b --out runs/4b/train_mixG.jsonl

Composition, the same at every model size:

    code         draft stop points, the shortest one per question     <runs>/verified_code_draft/train.jsonl
    give-up      code and math                                         <runs>/give_up_code, <runs>/give_up_math
    tool         stop points                                           <runs>/stop_points_tool/train.jsonl
    math         a third of the shortest whole correct solutions       <runs>/sft_shortest.jsonl
    instruction  stop points                                           <runs>/stop_points_constraint/train.jsonl

Math is deliberately thin and uses whole solutions, not stop points: stop data on math costs accuracy
on word problems (0.6B study), so the length pressure on math comes from the RL stage instead.
"""

import argparse
import json
from pathlib import Path
import random


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="Directory holding the per-domain outputs")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    runs = arguments.runs

    shortest = {}
    for r in load(runs / "verified_code_draft/train.jsonl"):
        if r["question_id"] not in shortest or r["tokens"] < shortest[r["question_id"]]["tokens"]:
            shortest[r["question_id"]] = r
    draft = list(shortest.values())
    give = load(runs / "give_up_code/train.jsonl") + load(runs / "give_up_math/train.jsonl")
    tool = load(runs / "stop_points_tool/train.jsonl")
    math = load(runs / "sft_shortest.jsonl")
    for r in math:
        r["kind"] = "math: whole shortest correct base solution"
    random.Random(0).shuffle(math)
    math = math[:len(math) // 3]
    constraint = load(runs / "stop_points_constraint/train.jsonl")

    rows = draft + give + tool + math + constraint
    arguments.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"mix G: {len(draft)} draft code (1/question) + {len(give)} give-up + {len(tool)} tool stop + "
          f"{len(math)} whole math (1/3) + {len(constraint)} constraint stop = {len(rows)} rows, "
          f"{sum(r['tokens'] for r in rows) / 1e6:.2f} M completion tokens, longest {max(r['tokens'] for r in rows)}")


if __name__ == "__main__":
    main()
