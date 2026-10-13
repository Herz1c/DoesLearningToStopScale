"""Checks this module against another implementation of the same protocol (here: the ThinkingCap
Test harness the Qwen3-0.6B study was scored with), row by row.

    python -m evaluation.crosscheck questions bench/questions.jsonl other/questions.jsonl
    python -m evaluation.crosscheck grades runs/x/records.jsonl other_run/records.jsonl

`questions`: are the two question files the same questions, prompts, answers and seeds?
`grades`: two gradings of the same stored replies; how many verdicts agree, per dataset, with
the disagreeing rows printed so each can be looked at.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def questions(ours, theirs):
    a, b = read(ours), read(theirs)
    print(f"{len(a)} vs {len(b)} questions; files identical: {Path(ours).read_bytes() == Path(theirs).read_bytes()}")
    keyed = {q["id"]: q for q in b}
    differ = Counter()
    for q in a:
        other = keyed.get(q["id"])
        if other is None:
            differ["missing in other"] += 1
            continue
        for field in set(q) | set(other):
            if q.get(field) != other.get(field):
                differ[f"{q['dataset']}:{field}"] += 1
    print("differences:", dict(differ) or "none")


def grades(ours, theirs):
    a, b = read(ours), read(theirs)
    keyed = {(r["id"], r["sample_index"]): r for r in b}
    agree, total, shown = Counter(), Counter(), defaultdict(list)
    for row in a:
        other = keyed[(row["id"], row["sample_index"])]
        if row.get("text") != other.get("text"):
            raise SystemExit(f"different replies for {row['id']}#{row['sample_index']}; not the same run")
        total[row["dataset"]] += 1
        if row["correct"] == other["correct"]:
            agree[row["dataset"]] += 1
        elif len(shown[row["dataset"]]) < 10:
            shown[row["dataset"]].append((row["id"], row["sample_index"], row["correct"], other["correct"],
                                          row.get("parse_error") or row.get("grading_error"),
                                          other.get("parse_error") or other.get("grading_error")))
    for dataset in sorted(total):
        acc_a = sum(bool(r["correct"]) for r in a if r["dataset"] == dataset) / total[dataset]
        acc_b = sum(bool(keyed[(r["id"], r["sample_index"])]["correct"]) for r in a
                    if r["dataset"] == dataset) / total[dataset]
        print(f"{dataset:<20} {agree[dataset]}/{total[dataset]} verdicts agree   "
              f"accuracy {acc_a:.4%} vs {acc_b:.4%}")
        for item in shown[dataset]:
            print("    ", item)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("questions", "grades"))
    parser.add_argument("ours")
    parser.add_argument("theirs")
    args = parser.parse_args()
    (questions if args.what == "questions" else grades)(args.ours, args.theirs)


if __name__ == "__main__":
    main()
