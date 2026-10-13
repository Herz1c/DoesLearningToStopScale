"""Difficulty-matched training pools (Amendment 2): mix easy and hard questions so that the base model's
share of rollouts that know the answer matches a target (the 0.6B base on its original pool).

    # 1. after rollouts on all easy and on 200 hard calibration questions:
    python stopping/mix_pools.py calibrate --easy runs/rollouts_gsm8k --hard runs/rollouts_math_hard --target 0.708
    # 2. after the hard rollouts are extended to n_hard questions:
    python stopping/mix_pools.py merge --easy runs/rollouts_gsm8k --hard runs/rollouts_math_hard --hard-fraction 0.5 \
        --out runs/rollouts_math_mix --pool-easy data/train.jsonl:gsm8k --pool-hard data/math_hard_train.jsonl \
        --pool-out data/math_mix_train.jsonl

calibrate: k = share of rollouts with `knows`; f = argmin over {0, 0.1, ..., 1} of |(1-f) k_easy + f k_hard - target|
(ties: the smaller f); prints f, the question counts for a pool of --size questions, and writes <hard>/calibration.json.
merge: the first (1-f)*size easy questions and the first f*size hard questions of the rollout files (pick order),
hard question ids offset by 1 000 000, into <out>/rollouts.jsonl; and a GRPO prompt pool file in which a uniform draw
is hard with probability f (all hard rows plus a Random(0) sample of easy rows, or the reverse if easy rows run short).
"""

import argparse
import json
from pathlib import Path
import random

OFFSET = 1_000_000


def load(directory):
    return [json.loads(line) for line in (Path(directory) / "rollouts.jsonl").read_text().splitlines()]


def in_order(rows):
    order = []
    for r in rows:
        if r["question_id"] not in order:
            order.append(r["question_id"])
    return order


def share(rows, key="knows"):
    return sum(bool(r[key]) for r in rows) / len(rows)


def profile(rows):
    return {"rollouts": len(rows), "questions": len(in_order(rows)), "knows": share(rows), "strict": share(rows, "correct"),
            "capped": sum(not r["finished"] for r in rows) / len(rows)}


def calibrate(arguments):
    easy, hard = profile(load(arguments.easy)), profile(load(arguments.hard))
    grid = [round(0.1 * i, 1) for i in range(11)]
    mixed = {f: (1 - f) * easy["knows"] + f * hard["knows"] for f in grid}
    f = min(grid, key=lambda g: (abs(mixed[g] - arguments.target), g))
    n_hard = round(f * arguments.size)
    result = {"target": arguments.target, "easy": easy, "hard": hard, "hard_fraction": f,
              "predicted_knows": mixed[f], "easy_questions": arguments.size - n_hard, "hard_questions": n_hard,
              "target_reached": abs(mixed[f] - arguments.target) <= 0.05}
    (Path(arguments.hard) / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def pool_rows(spec):
    path, _, source = spec.partition(":")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [r for r in rows if not source or r.get("source") == source]


def merge(arguments):
    f, size = arguments.hard_fraction, arguments.size
    n_hard = round(f * size)
    easy, hard = load(arguments.easy), load(arguments.hard)
    keep_easy = set(in_order(easy)[:size - n_hard])
    keep_hard = set(in_order(hard)[:n_hard])
    if len(keep_hard) < n_hard:
        raise SystemExit(f"only {len(keep_hard)} hard questions rolled out, {n_hard} needed")
    rows = [r for r in easy if r["question_id"] in keep_easy]
    rows += [dict(r, question_id=r["question_id"] + OFFSET) for r in hard if r["question_id"] in keep_hard]
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{out}/rollouts.jsonl: {len(keep_easy)} easy + {len(keep_hard)} hard questions, {len(rows)} rollouts; "
          f"profile {json.dumps(profile(rows))}")
    easy_pool, hard_pool = pool_rows(arguments.pool_easy), pool_rows(arguments.pool_hard)
    rng = random.Random(0)
    if f == 0:
        pool = easy_pool
    elif f == 1:
        pool = hard_pool
    else:
        want_easy = round(len(hard_pool) * (1 - f) / f)
        if want_easy <= len(easy_pool):
            pool = hard_pool + rng.sample(easy_pool, want_easy)
        else:
            pool = easy_pool + rng.sample(hard_pool, round(len(easy_pool) * f / (1 - f)))
    rng.shuffle(pool)
    Path(arguments.pool_out).write_text("".join(json.dumps(r) + "\n" for r in pool))
    n_h = sum(r in hard_pool for r in pool)
    print(f"{arguments.pool_out}: {len(pool)} prompts, {n_h / len(pool):.1%} hard")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    m = sub.add_parser("merge")
    for p in (c, m):
        p.add_argument("--easy", required=True)
        p.add_argument("--hard", required=True)
        p.add_argument("--size", type=int, default=1500)
    c.add_argument("--target", type=float, required=True)
    m.add_argument("--hard-fraction", type=float, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--pool-easy", required=True, help="path[:source] of the easy prompt rows")
    m.add_argument("--pool-hard", required=True)
    m.add_argument("--pool-out", required=True)
    arguments = parser.parse_args()
    (calibrate if arguments.command == "calibrate" else merge)(arguments)


if __name__ == "__main__":
    main()
