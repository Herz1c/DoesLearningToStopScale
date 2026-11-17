"""Training pools for Qwen3.8-27B, composed from the difficulty tiers by the calibration of 2026-11-17.

    python stopping/pools_27b.py        # writes data/pool27b_{math,code,tool,constraint}.jsonl

The Qwen3.8-27B base answered 200 questions x 4 of every tier (chains/calibrate_27b.sh; the code tiers were
graded afterwards, on a machine with a sandbox). The share of rollouts that know the answer, against the share
the Qwen3-0.6B base had on its own pools (the anchor):

    math        anchor 70.8%   GSM8K 96.1%, MATH levels 3-4 92.2%, level 5 64.0% (38% run out of tokens)
    code        anchor 51.9%   KodCode easy 85.8%, hard with GPT pass rate above 0.12 70.2%, at most 0.12 59.6%
    tool        anchor 63.7%   Glaive 90.2%, wide 87.9%, removed 83.8%
    constraint  anchor 66.6%   1-2 constraints 93.6%, three 80.1%, four 56.0%

Pools, each a single file that stopping/generate_rollouts.py and stopping/train_grpo.py read directly:

    math        300 MATH levels 3-4 + 1 200 level 5 (20% / 80%, predicted 69.7%).
    code        the 4B pool again: the 492 hardest tasks first, then the 1 008 others (predicted 66.7%). The
                anchor is out of reach: the hardest tier alone is still above it, and it is too small for a pool.
    tool        the removed tier, 1 500 prompts. The anchor is out of reach (the domain is nearly solved for
                this model, as at 4B); the rule then takes the hardest tier alone.
    constraint  640 prompts with three constraints + 960 with four (40% / 60%, predicted 65.7%).

Questions come in the pick order of stopping/generate_rollouts.py, so the calibration questions are the
first of each part.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.generate_rollouts import pick  # noqa: E402

SIZE = 1500


def ordered(path, sources):
    """All rows of a pool file in the order generate_rollouts.py would take them, without the ids."""
    rows = pick(path, set(sources), 10 ** 9, 0)
    return [{k: v for k, v in r.items() if k != "question_id"} for r in rows]


def write(name, rows):
    Path(f"data/pool27b_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    parts = {}
    for r in rows:
        parts[r["part"]] = parts.get(r["part"], 0) + 1
    assert len({json.dumps(r["prompt"]) for r in rows}) == len(rows), f"{name}: repeated prompts"
    print(f"data/pool27b_{name}.jsonl: {len(rows)} rows {parts}")


def tagged(rows, part):
    return [dict(r, part=part) for r in rows]


def main():
    # math: 20% MATH levels 3-4, 80% level 5 (the calibration file first, then the level-5 rows of the 1.7B pool)
    n_hard = round(0.8 * SIZE)
    level5 = ordered("data/math_l5_train.jsonl", ["math"])
    level5 += [r for r in ordered("data/math_hard_train.jsonl", ["math"]) if r["level"] == "Level 5"]
    # 12 of these problems are also validation questions (data/val.jsonl has them through Skywork-OR1): left out
    held = {json.dumps(json.loads(line)["prompt"]) for name in ("val", "math_hard_val")
            for line in Path(f"data/{name}.jsonl").read_text().splitlines()}
    level5 = [r for r in level5 if json.dumps(r["prompt"]) not in held]
    level34 = [r for r in ordered("data/math_l34_train.jsonl", ["math"]) if json.dumps(r["prompt"]) not in held]
    assert len(level5) >= n_hard, len(level5)
    write("math", tagged(level34[:SIZE - n_hard], "math_l34")
          + tagged(level5[:n_hard], "math_l5"))

    # code: the hardest tier whole, filled up with the tier below (together: the 4B pool)
    hardest = ordered("data/code27_t2.jsonl", ["kodcode"])
    write("code", tagged(hardest, "t2") + tagged(ordered("data/code27_t1.jsonl", ["kodcode"])[:SIZE - len(hardest)], "t1"))

    # tool: the hardest tier alone (targets corrected by tool_tasks.py relabel)
    write("tool", tagged(ordered("data/tool_removed_fixed.jsonl", ["glaive_removed"])[:SIZE], "removed"))

    # constraint: 40% three constraints, 60% four
    write("constraint", tagged(ordered("data/constraint_c3.jsonl", ["dolly_constraints"])[:640], "c3")
          + tagged(ordered("data/constraint_c4.jsonl", ["dolly_constraints"])[:960], "c4"))


if __name__ == "__main__":
    main()
