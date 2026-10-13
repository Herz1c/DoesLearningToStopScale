"""Grade a rollout file again with the current graders (stopping/grading.py), e.g. after a grader fix.

    python stopping/regrade_rollouts.py runs/rollouts_code              # rewrites rollouts.jsonl, keeps a .bak
    python stopping/regrade_rollouts.py runs/rollouts_code --dry-run    # only reports what would change

Rewrites the `correct` (strict) and `knows` (lenient) fields of every rollout and prints how many changed.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.grading import grade  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    path = arguments.rollouts / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    verdicts = grade([(r, r["text"]) for r in rows])
    changes = Counter()
    for r, (strict, knows) in zip(rows, verdicts):
        changes["strict " + ("same" if bool(r["correct"]) == strict else f"{bool(r['correct'])}->{strict}")] += 1
        changes["knows " + ("same" if bool(r["knows"]) == knows else f"{bool(r['knows'])}->{knows}")] += 1
        r["correct"], r["knows"] = strict, knows
    n = len(rows)
    print(f"{n} rollouts: strict {sum(r['correct'] for r in rows) / n:.1%}, knows {sum(r['knows'] for r in rows) / n:.1%}")
    for key, count in sorted(changes.items()):
        print(f"  {key:22s} {count}")
    if not arguments.dry_run:
        shutil.copy(path, path.with_suffix(".jsonl.bak"))
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"rewrote {path} (backup {path.with_suffix('.jsonl.bak').name})")


if __name__ == "__main__":
    main()
