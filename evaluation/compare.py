"""Paired comparison of two runs of evaluation.run on the same questions.

    python -m evaluation.compare runs/base runs/mine [--json runs/mine.compare.json]

Replies are paired on (question, sample); the two runs drew them with the same seeds, so pairing
removes question difficulty from the difference. Reported per dataset and as the average over
the four benchmarks (BFCL's three categories pooled into one): accuracy change = arithmetic mean
of the benchmarks' changes, tokens saved = 1 - geometric mean of their token ratios.
Refuses runs that differ in questions, samples, seed or token caps.
"""

import argparse
import json
from pathlib import Path

from evaluation import protocol
from evaluation.stats import macro, paired, summarize


def load(run):
    run = Path(run)
    meta = json.loads((run / "results.json").read_text())
    rows = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines() if line.strip()]
    return meta, rows


def setup(meta):
    """The settings pairing depends on. Reads runs of this module and of the reference harness."""
    return {"questions": meta.get("questions_sha256") or meta.get("cohort_sha256"),
            "samples": meta["samples"], "seed": meta["seed"],
            "caps": meta.get("caps") or meta.get("max_tokens")}


def compare(base_dir, new_dir):
    base_meta, base_rows = load(base_dir)
    new_meta, new_rows = load(new_dir)
    a, b = setup(base_meta), setup(new_meta)
    if a != b:
        raise SystemExit("runs are not comparable: " + ", ".join(
            f"{k} {a[k]} vs {b[k]}" for k in a if a[k] != b[k]))
    index = {(r["id"], r["sample_index"]): r for r in new_rows}
    datasets, pooled = {}, {}
    for dataset in sorted({r["dataset"] for r in base_rows}):
        left = [r for r in base_rows if r["dataset"] == dataset and (r["id"], r["sample_index"]) in index]
        right = [index[(r["id"], r["sample_index"])] for r in left]
        if not left:
            continue
        before, after = summarize(left), summarize(right)
        datasets[dataset] = {"base": before, "new": after,
                             "accuracy_delta": after["accuracy"] - before["accuracy"],
                             "token_reduction": 1 - after["total_output_tokens"] / before["total_output_tokens"],
                             "interval": paired(left, right)}
        group = pooled.setdefault(protocol.benchmark_of(dataset), ([], []))
        group[0].extend(left)
        group[1].extend(right)
    benchmarks = {}
    for name, (left, right) in pooled.items():
        before, after = summarize(left), summarize(right)
        benchmarks[name] = {"accuracy_delta": after["accuracy"] - before["accuracy"],
                            "token_ratio": after["total_output_tokens"] / before["total_output_tokens"]}
    average = None
    if len(benchmarks) > 1:
        ratios = [v["token_ratio"] for v in benchmarks.values()]
        product = 1.0
        for r in ratios:
            product *= r
        average = {"benchmarks": sorted(benchmarks),
                   "accuracy_delta": sum(v["accuracy_delta"] for v in benchmarks.values()) / len(benchmarks),
                   "token_reduction": 1 - product ** (1 / len(ratios)),
                   "interval": macro(list(pooled.values()))}
    return {"base": str(base_dir), "new": str(new_dir), "base_model": [base_meta.get("model"), base_meta.get("adapter")],
            "new_model": [new_meta.get("model"), new_meta.get("adapter")], "setup": a,
            "datasets": datasets, "benchmarks": benchmarks, "average": average}


def report(c):
    lines = [f"base  {c['base']}  ({c['base_model'][0]} + {c['base_model'][1]})",
             f"new   {c['new']}  ({c['new_model'][0]} + {c['new_model'][1]})", ""]
    for name, d in c["datasets"].items():
        lines.append(f"{name}  ({d['interval'].get('questions', '?')} questions x {c['setup']['samples']})")
        lines += block(d)
    if c["average"]:
        lines.append(f"\naverage over {len(c['average']['benchmarks'])} benchmarks "
                     f"({', '.join(c['average']['benchmarks'])}; BFCL pooled; geometric mean of token ratios)")
        lines += block(c["average"])
    return "\n".join(lines)


def block(d):
    tr, ad = d["interval"]["token_reduction"], d["interval"]["accuracy_delta"]
    out = []
    if tr:
        out.append(f"  tokens saved     {d['token_reduction']:+7.1%}    CI [{tr[0]:+.1%}, {tr[1]:+.1%}]")
    if ad:
        out.append(f"  accuracy change  {d['accuracy_delta'] * 100:+6.2f} pp  CI [{ad[0] * 100:+.2f} pp, {ad[1] * 100:+.2f} pp]")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--json")
    args = parser.parse_args()
    comparison = compare(args.base, args.new)
    if args.json:
        Path(args.json).write_text(json.dumps(comparison, indent=2) + "\n")
    print(report(comparison))


if __name__ == "__main__":
    main()
