"""Summary numbers and bootstrap intervals. Resampling is always over questions, never over single
generations: the 8 samples of one question are one question's worth of evidence.

    summarize(rows)                 accuracy, truncation, mean tokens, ... of one benchmark
    interval(rows)                  95% CI of accuracy and mean tokens
    paired(base_rows, new_rows)     95% CI of the accuracy change and of tokens saved
    macro(groups)                   the same for the average over benchmarks: arithmetic mean of
                                    accuracy changes, 1 - geometric mean of token ratios, each
                                    benchmark resampled on its own

Tokens saved on one benchmark = 1 - sum(new tokens) / sum(base tokens), over all replies, capped
ones included; a ratio of sums, so long replies weigh in proportion to their length.
"""

import numpy as np

RESAMPLES = 2000
SEED = 42


def summarize(rows):
    graded = [r for r in rows if r["correct"] is not None]
    if not graded:
        raise ValueError("no graded rows")
    n = len(graded)
    return {"n": n, "prompts": len({r["id"] for r in graded}), "ungraded": len(rows) - n,
            "accuracy": sum(bool(r["correct"]) for r in graded) / n,
            "truncation_rate": sum(bool(r["truncated"]) for r in graded) / n,
            "parse_failure_rate": sum(bool(r.get("parse_error")) for r in graded) / n,
            "mean_output_tokens": sum(r["output_tokens"] for r in graded) / n,
            "mean_reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in graded) / n,
            "total_output_tokens": sum(r["output_tokens"] for r in graded)}


def per_question(items, question, columns):
    """Items -> array (questions x columns) of per-question sums, in first-seen order."""
    sums = {}
    for item in items:
        values = columns(item)
        if values is None:
            continue
        total = sums.setdefault(question(item), [0.0] * len(values))
        for i, v in enumerate(values):
            total[i] += v
    return np.array(list(sums.values()), dtype=float)


def draws(table, rng, resamples=RESAMPLES):
    """Bootstrap column sums: (resamples x columns), each drawing len(table) questions with replacement."""
    picks = rng.integers(0, len(table), size=(resamples, len(table)))
    return table[picks].sum(axis=1)


def ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def interval(rows, resamples=RESAMPLES, seed=SEED):
    table = per_question(rows, lambda r: r["id"], lambda r: None if r["correct"] is None else
                         (float(r["correct"]), r["output_tokens"], 1))
    if len(table) < 2:
        return {"accuracy": None, "mean_output_tokens": None}
    totals = draws(table, np.random.default_rng(seed), resamples)
    return {"accuracy": ci(totals[:, 0] / totals[:, 2]),
            "mean_output_tokens": ci(totals[:, 1] / totals[:, 2]), "resamples": resamples}


def pair_table(base, new):
    """Aligned row lists -> per-question (base correct, new correct, base tokens, new tokens, count)."""
    pairs = [(b, c) for b, c in zip(base, new) if b["correct"] is not None and c["correct"] is not None]
    return per_question(pairs, lambda p: p[0]["id"],
                        lambda p: (float(p[0]["correct"]), float(p[1]["correct"]),
                                   p[0]["output_tokens"], p[1]["output_tokens"], 1))


def paired(base, new, resamples=RESAMPLES, seed=SEED):
    table = pair_table(base, new)
    if len(table) < 2:
        return {"accuracy_delta": None, "token_reduction": None}
    t = draws(table, np.random.default_rng(seed), resamples)
    return {"accuracy_delta": ci((t[:, 1] - t[:, 0]) / t[:, 4]),
            "token_reduction": ci(1 - t[:, 3] / t[:, 2]), "resamples": resamples,
            "questions": len(table)}


def macro(groups, resamples=RESAMPLES, seed=SEED):
    """groups: [(base rows, new rows), ...], one per benchmark."""
    tables = [pair_table(b, n) for b, n in groups]
    if not tables or any(len(t) < 2 for t in tables):
        return {"accuracy_delta": None, "token_reduction": None}
    rng = np.random.default_rng(seed)
    deltas, log_ratios = [], []
    for table in tables:
        t = draws(table, rng, resamples)
        deltas.append((t[:, 1] - t[:, 0]) / t[:, 4])
        log_ratios.append(np.log(t[:, 3] / t[:, 2]))
    return {"accuracy_delta": ci(np.mean(deltas, axis=0)),
            "token_reduction": ci(1 - np.exp(np.mean(log_ratios, axis=0))), "resamples": resamples}
