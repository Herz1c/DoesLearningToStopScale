"""Training pools for Qwen3-4B, composed from the difficulty tiers by the calibration of 2026-11-13.

    python stopping/pools_4b.py        # writes data/pool4b_{math,code,tool,constraint}.jsonl

The Qwen3-4B base answered 200 questions x 4 of every tier (chains/calibrate_4b.sh). The share of
rollouts that know the answer, against the share the Qwen3-0.6B base had on its own pools (the anchor):

    math        anchor 70.8%   GSM8K 90.2%, MATH levels 3-4 62.2%, level 5 15.8% (84% run out of tokens)
    code        anchor 51.9%   KodCode easy 79.4%, medium 58.8%, hard 51.6-57.5%
    tool        anchor 63.7%   Glaive 95.2%, wide 84.8%, removed 88.8%   (corrected targets)
    constraint  anchor 66.6%   1-2 constraints 91.4%, three 78.4%, four 66.4%   (corrected checker)

Pools, each a single file that stopping/generate_rollouts.py and stopping/train_grpo.py read directly:

    math        450 GSM8K + 1 050 MATH levels 3-4 (30% / 70%, predicted 70.6%). Level 5 is left out: the
                model seldom finishes those within the token cap, so they give no stopping point.
    code        every hard task (1 271) + 229 medium tasks (predicted about 54%).
    tool        750 wide + 750 removed. The anchor is out of reach (the domain is nearly solved for this
                model); the removed tier is kept for its tasks where declining is the right answer.
    constraint  the four-constraint tier, 1 600 prompts.

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
    Path(f"data/pool4b_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    parts = {}
    for r in rows:
        parts[r["part"]] = parts.get(r["part"], 0) + 1
    print(f"data/pool4b_{name}.jsonl: {len(rows)} rows {parts}")


def tagged(rows, part):
    return [dict(r, part=part) for r in rows]


def main():
    # math: 30% GSM8K, 70% MATH levels 3-4 (first from the 1.7B pool, then from the extra file)
    hard = [r for r in ordered("data/math_hard_train.jsonl", ["math"]) if r["level"] in ("Level 3", "Level 4")]
    hard += ordered("data/math_l34_train.jsonl", ["math"])
    n_hard = round(0.7 * SIZE)
    assert len(hard) >= n_hard, len(hard)
    write("math", tagged(ordered("data/train.jsonl", ["gsm8k"])[:SIZE - n_hard], "gsm8k") + tagged(hard[:n_hard], "math_l34"))

    # code: every hard task, filled up with medium ones
    from datasets import load_dataset
    level = {r["question_id"]: r["gpt_difficulty"] for r in load_dataset("KodCode/KodCode-Light-RL-10K", split="train")}
    mixed = ordered("data/code_hard_train.jsonl", ["kodcode"])
    hard = [r for r in mixed if level[r["kodcode_id"]] == "hard"] + ordered("data/code_hardest_train.jsonl", ["kodcode"])
    medium = [r for r in mixed if level[r["kodcode_id"]] == "medium"]
    assert all(level[r["kodcode_id"]] == "hard" for r in hard) and len({r["kodcode_id"] for r in hard}) == len(hard)
    write("code", tagged(hard[:SIZE], "hard") + tagged(medium[:max(0, SIZE - len(hard))], "medium"))

    # tool: half wide, half removed (targets corrected by tool_tasks.py relabel)
    write("tool", tagged(ordered("data/tool_wide_fixed.jsonl", ["glaive_wide"])[:SIZE // 2], "wide")
          + tagged(ordered("data/tool_removed_fixed.jsonl", ["glaive_removed"])[:SIZE // 2], "removed"))

    # constraint: the four-constraint tier
    write("constraint", tagged(ordered("data/constraint_c4.jsonl", ["dolly_constraints"]), "c4"))


if __name__ == "__main__":
    main()
