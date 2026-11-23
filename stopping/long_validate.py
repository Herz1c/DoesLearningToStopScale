"""Validation of long-reasoning adapters against the base model, and the rule that picks between them.

    python stopping/long_validate.py runs/27b_long_val/base trace=runs/27b_long_val/trace decision=runs/27b_long_val/decision \
        --out runs/27b_long_val/pilot.json

Every directory holds rollouts of the same validation questions (stopping/long_rollouts.py, same data file, 4 per
question). The rule was fixed before any training (PROGRESS.md 9d, 9i):

  accuracy = the strict grade; a reply that ran into the token cap is wrong.
  per domain: change in accuracy (pp) and saving = 1 - mean thinking tokens / the base's mean thinking tokens;
  overall = the unweighted mean over the domains, for both.
  length buckets by the base's mean thinking per question, all domains together: < 2k, 2-8k, 8-32k, > 32k; a
  bucket with fewer than 20 questions joins the next shorter one (the shortest joins the next longer);
  bucket change = mean over its questions of the question's change in accuracy.
  An adapter PASSES when overall change >= -1.0 pp and every bucket >= -3.0 pp.
  The winner is the passing adapter with the larger overall saving; "none" when none passes.

Point estimates decide; 95% intervals (bootstrap over questions within domains, 2000 draws) are printed next to
them. The last line is "winner: <name>"; --out also gets everything as JSON. Exit code 0 either way.
"""

import argparse
import collections
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.readouts import read  # noqa: E402

BUCKETS = ((0, 2000, "< 2k"), (2000, 8000, "2-8k"), (8000, 32000, "8-32k"), (32000, 10 ** 9, "> 32k"))
OVERALL, BUCKET, SMALL = -1.0, -3.0, 20


def domain_of(row):
    if row.get("domain"):
        return row["domain"]
    part = row.get("part", "")
    return ("math" if part.startswith(("dm_", "harp")) else "code" if part.startswith("cf_") else
            "choice" if part.startswith(("medqa", "easy")) else "constraint" if row["task"] == "constraint" else "tool")


def questions(directory):
    """question id -> {"domain", "right": [0/1 ...], "thinking": [...]} of one run."""
    table = {}
    for r in read(directory, "rollouts*.jsonl"):
        q = table.setdefault(r["question_id"], {"domain": domain_of(r), "right": [], "thinking": []})
        q["right"].append(float(bool(r["correct"]) and bool(r["finished"])))
        q["thinking"].append(r["thinking_tokens"])
    return table


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summary(base, run, ids):
    """Overall change (pp), overall saving, per-domain rows for the question ids given."""
    domains = collections.defaultdict(list)
    for q in ids:
        domains[base[q]["domain"]].append(q)
    rows = {}
    for d, group in sorted(domains.items()):
        accuracy_base = mean(mean(base[q]["right"]) for q in group)
        accuracy = mean(mean(run[q]["right"]) for q in group)
        think_base = mean(t for q in group for t in base[q]["thinking"])
        think = mean(t for q in group for t in run[q]["thinking"])
        rows[d] = {"questions": len(group), "accuracy_base": 100 * accuracy_base, "accuracy": 100 * accuracy,
                   "change": 100 * (accuracy - accuracy_base), "thinking_base": think_base, "thinking": think,
                   "saving": 100 * (1 - think / max(1.0, think_base))}
    return mean(r["change"] for r in rows.values()), mean(r["saving"] for r in rows.values()), rows


def buckets(base, run, ids):
    groups = [[q for q in ids if low <= mean(base[q]["thinking"]) < high] for low, high, _ in BUCKETS]
    names = [name for _, _, name in BUCKETS]
    k = len(groups) - 1
    while k > 0:                                    # small buckets join the next shorter one
        if 0 < len(groups[k]) < SMALL:
            groups[k - 1] += groups[k]
            names[k - 1] += " + " + names[k]
            groups[k] = []
        k -= 1
    kept = [(n, g) for n, g in zip(names, groups) if g]
    if len(kept) > 1 and len(kept[0][1]) < SMALL:   # the shortest joins the next longer
        kept[1] = (kept[0][0] + " + " + kept[1][0], kept[0][1] + kept[1][1])
        kept = kept[1:]
    return [{"bucket": n, "questions": len(g),
             "change": 100 * mean(mean(run[q]["right"]) - mean(base[q]["right"]) for q in g)} for n, g in kept]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidates", nargs="+", help="name=directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    arguments = parser.parse_args()

    base = questions(arguments.base)
    report, passing = {"rule": {"overall_pp": OVERALL, "bucket_pp": BUCKET}, "candidates": {}}, []
    for item in arguments.candidates:
        name, directory = item.split("=", 1)
        run = questions(directory)
        ids = sorted(q for q in base if q in run)
        if not ids:
            print(f"{name}: no questions in common with the base ({directory})")
            continue
        change, saving, rows = summary(base, run, ids)
        rng, draws = random.Random(0), []
        by_domain = collections.defaultdict(list)
        for q in ids:
            by_domain[base[q]["domain"]].append(q)
        for _ in range(arguments.draws):
            sample = [q for group in by_domain.values() for q in rng.choices(group, k=len(group))]
            draws.append(summary(base, run, sample)[:2])
        low, high = int(0.025 * len(draws)), int(0.975 * len(draws)) - 1
        change_ci = [sorted(d[0] for d in draws)[k] for k in (low, high)]
        saving_ci = [sorted(d[1] for d in draws)[k] for k in (low, high)]
        parts = buckets(base, run, ids)
        short_base = mean(t < 20 for q in ids for t in base[q]["thinking"])
        short = mean(t < 20 for q in ids for t in run[q]["thinking"])
        passed = change >= OVERALL and all(b["change"] >= BUCKET for b in parts)
        if passed:
            passing.append((saving, name))
        report["candidates"][name] = {"questions": len(ids), "change": change, "change_ci": change_ci, "saving": saving,
                                      "saving_ci": saving_ci, "domains": rows, "buckets": parts, "passed": passed,
                                      "under_20_thinking_tokens": [100 * short_base, 100 * short]}
        print(f"\n== {name}: {len(ids)} questions ({len(base) - len(ids)} of the base's missing)")
        print(f"{'domain':12} {'questions':>9} {'base acc':>9} {'acc':>7} {'change':>8} {'base think':>11} {'think':>8} {'saving':>8}")
        for d, r in rows.items():
            print(f"{d:12} {r['questions']:9} {r['accuracy_base']:8.1f}% {r['accuracy']:6.1f}% {r['change']:+7.1f}  "
                  f"{r['thinking_base']:11.0f} {r['thinking']:8.0f} {r['saving']:7.1f}%")
        print(f"overall: change {change:+.2f} pp [{change_ci[0]:+.2f}, {change_ci[1]:+.2f}], "
              f"saving {saving:.1f}% [{saving_ci[0]:.1f}, {saving_ci[1]:.1f}]   (rule: change >= {OVERALL})")
        for b in parts:
            mark = "" if b["change"] >= BUCKET else "   <-- below the rule"
            print(f"  bucket {b['bucket']:22} {b['questions']:4} questions  change {b['change']:+6.2f} pp{mark}")
        print(f"  replies with under 20 thinking tokens: base {100 * short_base:.1f}%, {name} {100 * short:.1f}%")
        print(f"  {name}: {'PASSES' if passed else 'DOES NOT PASS'} the rule")
    winner = max(passing)[1] if passing else "none"
    report["winner"] = winner
    if arguments.out:
        arguments.out.write_text(json.dumps(report, indent=2))
    print(f"\nwinner: {winner}")


if __name__ == "__main__":
    main()
