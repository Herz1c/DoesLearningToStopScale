"""Hard-tail check (Amendment 2): does the shorter model give up accuracy on the questions that are hard for the base?

    python analysis/hard_tail.py runs/base runs/grpo_full1                     # evaluation.run directories
    python analysis/hard_tail.py runs/heldout/base runs/heldout/grpo           # analysis/heldout_eval.py directories

For every benchmark (BFCL pooled), two subsets of questions defined by the *base* run only:
- long:  the fifth of questions with the highest mean base reply length;
- hard:  questions the base answers correctly in at most half of its samples.
Reported per subset: base and new accuracy, accuracy change and tokens saved with paired question-level bootstrap
intervals (evaluation/stats.py), next to the same numbers on all questions.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.protocol import benchmark_of  # noqa: E402
from evaluation.stats import paired  # noqa: E402


def load(directory):
    """-> {(group, id, sample): row with id / correct / output_tokens}, for either run format."""
    rows = {}
    for line in (Path(directory) / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "dataset" in r:
            group, sample, tokens = benchmark_of(r["dataset"]), r["sample_index"], r["output_tokens"]
        else:
            group, sample, tokens = r["set"], r["sample"], r["tokens"]
        rows[(group, r["id"], sample)] = {"id": f"{group}:{r['id']}", "correct": r["correct"], "output_tokens": tokens}
    return rows


def report(name, base_rows, new_rows):
    if not base_rows:
        return
    acc = lambda rows: sum(bool(r["correct"]) for r in rows) / len(rows)
    saved = 1 - sum(r["output_tokens"] for r in new_rows) / sum(r["output_tokens"] for r in base_rows)
    ci = paired(base_rows, new_rows)
    q = len({r["id"] for r in base_rows})
    tr, ad = ci["token_reduction"] or [float("nan")] * 2, ci["accuracy_delta"] or [float("nan")] * 2
    print(f"  {name:6s} {q:5d} q  acc {acc(base_rows):6.1%} -> {acc(new_rows):6.1%}  "
          f"Δ {100 * (acc(new_rows) - acc(base_rows)):+6.2f} pp [{100 * ad[0]:+.2f}, {100 * ad[1]:+.2f}]  "
          f"saved {saved:+6.1%} [{tr[0]:+.1%}, {tr[1]:+.1%}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("new")
    arguments = parser.parse_args()
    base, new = load(arguments.base), load(arguments.new)
    keys = [k for k in base if k in new and base[k]["correct"] is not None and new[k]["correct"] is not None]
    for group in sorted({k[0] for k in keys}):
        mine = [k for k in keys if k[0] == group]
        per_q = defaultdict(list)
        for k in mine:
            per_q[k[1]].append(base[k])
        length = {q: sum(r["output_tokens"] for r in v) / len(v) for q, v in per_q.items()}
        solved = {q: sum(bool(r["correct"]) for r in v) / len(v) for q, v in per_q.items()}
        cut = sorted(length.values(), reverse=True)[max(0, len(length) // 5 - 1)]
        subsets = {"all": lambda q: True, "long": lambda q: length[q] >= cut, "hard": lambda q: solved[q] <= 0.5}
        print(f"{group}")
        for name, keep in subsets.items():
            chosen = [k for k in mine if keep(k[1])]
            report(name, [base[k] for k in chosen], [new[k] for k in chosen])


if __name__ == "__main__":
    main()
