"""Accuracy change and tokens saved per difficulty bin, comparable across model sizes (Amendment 3).

    # main benchmarks, bins = how often a fixed reference model (0.6B base, own-module run) solved the question
    python analysis/difficulty_bins.py runs/base runs/grpo_full1 --reference runs/verify/base_06b
    # held-out MATH-500, bins = MATH level 1-5, graded leniently (knows): base models close MATH answers with
    # \\boxed{} instead of the requested "Final answer:" line, so strict accuracy would measure format only
    python analysis/difficulty_bins.py runs/heldout/base runs/heldout/grpo --levels --metric knows

The bins are defined independently of the two runs compared (a fixed reference run, or the dataset's own level), so
the same bins can be laid over 0.6B, 1.7B and larger models and the curves compared; for the reference model itself,
use a second, independent run of it as `base` to avoid regression to the mean. Reference bins: 0/8, 1-4/8, 5-7/8,
8/8 samples correct. Intervals: paired question-level bootstrap (evaluation/stats.py). --json writes the table.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.protocol import benchmark_of  # noqa: E402
from evaluation.stats import paired  # noqa: E402

REFERENCE_BINS = (("0/8", 0, 0), ("1-4/8", 1, 4), ("5-7/8", 5, 7), ("8/8", 8, 8))


def load(directory, metric):
    rows = {}
    for line in (Path(directory) / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "dataset" in r:
            group, sample, tokens = benchmark_of(r["dataset"]), r["sample_index"], r["output_tokens"]
        else:
            group, sample, tokens = r["set"], r["sample"], r["tokens"]
        value = r.get(metric) if metric in r else r["correct"]
        rows[(group, r["id"], sample)] = {"id": f"{group}:{r['id']}", "correct": value, "output_tokens": tokens,
                                          "level": r.get("level")}
    return rows


def summary(base_rows, new_rows):
    acc = lambda rows: sum(bool(r["correct"]) for r in rows) / len(rows)
    ci = paired(base_rows, new_rows)
    return {"questions": len({r["id"] for r in base_rows}), "base_acc": acc(base_rows), "new_acc": acc(new_rows),
            "acc_delta": acc(new_rows) - acc(base_rows), "acc_delta_ci": ci["accuracy_delta"],
            "base_mean_tokens": sum(r["output_tokens"] for r in base_rows) / len(base_rows),
            "tokens_saved": 1 - sum(r["output_tokens"] for r in new_rows) / sum(r["output_tokens"] for r in base_rows),
            "tokens_saved_ci": ci["token_reduction"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--reference", help="Run whose per-question solve rate defines the bins")
    parser.add_argument("--levels", action="store_true", help="Bins = the rows' `level` field (MATH-500)")
    parser.add_argument("--metric", default="correct", choices=("correct", "knows"))
    parser.add_argument("--json")
    arguments = parser.parse_args()
    base, new = load(arguments.base, arguments.metric), load(arguments.new, arguments.metric)
    keys = [k for k in base if k in new and base[k]["correct"] is not None and new[k]["correct"] is not None]

    if arguments.reference:
        reference = load(arguments.reference, "correct")
        solved = defaultdict(int)
        for (group, qid, _), r in reference.items():
            solved[(group, qid)] += bool(r["correct"])
        def bin_of(k):
            n = solved.get((k[0], k[1]))
            return None if n is None else next(name for name, lo, hi in REFERENCE_BINS if lo <= n <= hi)
        order = [name for name, _, _ in REFERENCE_BINS]
    elif arguments.levels:
        bin_of = lambda k: f"level {base[k]['level']}" if base[k]["level"] is not None else None
        order = [f"level {i}" for i in range(1, 6)]
    else:
        raise SystemExit("give --reference RUN or --levels")

    table = {}
    for group in sorted({k[0] for k in keys}):
        mine = [k for k in keys if k[0] == group]
        table[group] = {"all": summary([base[k] for k in mine], [new[k] for k in mine])}
        for name in order:
            chosen = [k for k in mine if bin_of(k) == name]
            if len({k[1] for k in chosen}) >= 2:
                table[group][name] = summary([base[k] for k in chosen], [new[k] for k in chosen])
    for group, bins in table.items():
        print(group)
        for name, s in bins.items():
            a, t = s["acc_delta_ci"] or [float("nan")] * 2, s["tokens_saved_ci"] or [float("nan")] * 2
            print(f"  {name:9s} {s['questions']:4d} q  base {s['base_acc']:6.1%} ({s['base_mean_tokens']:5.0f} tok)  "
                  f"Δacc {100 * s['acc_delta']:+6.2f} pp [{100 * a[0]:+.2f}, {100 * a[1]:+.2f}]  "
                  f"saved {s['tokens_saved']:+6.1%} [{t[0]:+.1%}, {t[1]:+.1%}]")
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
