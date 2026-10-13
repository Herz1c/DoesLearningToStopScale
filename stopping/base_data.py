"""Writes data/train.jsonl and data/val.jsonl: math and multiple-choice training questions.

    python stopping/base_data.py

Sources (Hugging Face): openai/gsm8k (main, train), Skywork/Skywork-OR1-RL-Data (math, integer
answers, the easy band: DeepSeek-R1-Distill-Qwen-1.5B failed at most 8 of 16 tries),
tau/commonsense_qa (train), allenai/ai2_arc (ARC-Easy, train). Each source is shuffled with one
Random(0), its first 200 go to val, the rest to train, and train is shuffled once more. The recipe
uses only the GSM8K rows of train.jsonl; the others are kept so line numbers (= question ids in
every rollout file) stay the same as in the Qwen3-0.6B study, whose file this reproduces
byte for byte (sha256 in README).

Row: {"prompt": [{"role": "user", "content": question + instruction}], "answer": "42",
      "task": "math" | "multiple_choice", "source": "gsm8k" | ...}
"""

import json
from pathlib import Path
import random
import re

from datasets import load_dataset

MATH = "Put the final answer on the last line, like: Final answer: 42"
CHOICE = "Answer with the label of the correct option. Put it on the last line, like: Final answer: B"
VAL = 200


def make(question, instruction, answer, task, source):
    return {"prompt": [{"role": "user", "content": f"{question}\n\n{instruction}"}],
            "answer": answer, "task": task, "source": source}


def options(question, choices):
    return question + "\n\n" + "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))


def gsm8k():
    return [make(r["question"], MATH, r["answer"].split("####")[-1].strip().replace(",", ""), "math", "gsm8k")
            for r in load_dataset("openai/gsm8k", "main", split="train")]


def skywork():
    rows = []
    for r in load_dataset("Skywork/Skywork-OR1-RL-Data", split="math"):
        gold = json.loads(r["reward_model"]["ground_truth"])
        easy = r["extra_info"]["model_difficulty"]["DeepSeek-R1-Distill-Qwen-1.5B"] <= 8
        if len(gold) == 1 and re.fullmatch(r"-?\d+", gold[0].strip()) and easy:
            rows.append(make(r["prompt"][-1]["content"], MATH, gold[0].strip(), "math", "skywork"))
    return rows


def commonsense_qa():
    return [make(options(r["question"], r["choices"]), CHOICE, r["answerKey"], "multiple_choice", "commonsense_qa")
            for r in load_dataset("tau/commonsense_qa", split="train")]


def arc_easy():
    return [make(options(r["question"], r["choices"]), CHOICE, r["answerKey"], "multiple_choice", "arc_easy")
            for r in load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")]


def main():
    rng = random.Random(0)
    train, val = [], []
    for source in (gsm8k, skywork, commonsense_qa, arc_easy):
        rows = source()
        rng.shuffle(rows)
        val += rows[:VAL]
        train += rows[VAL:]
    rng.shuffle(train)
    Path("data").mkdir(exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        Path(f"data/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"data/{name}.jsonl: {len(rows)} rows")


if __name__ == "__main__":
    main()
