"""Scaling report: the same method at two model sizes, side by side (tables + figures).

    python analysis/scaling_report.py --out figures/scaling \
        --size "0.6B=/home/.../runs/base:/home/.../runs/grpo_submission,/home/.../runs/grpo_full" \
        --size "1.7B=runs/base:runs/grpo_full1,runs/grpo_full2" \
        --ladder "0.6B=runs/heldout06/base_math500:runs/heldout06/grpo_math500" \
        --ladder "1.7B=runs/heldout/base_math500:runs/heldout/grpo" \
        --reference runs/verify/base_06b

For each size: the main benchmarks (tokens saved, accuracy change, per benchmark and averaged; with two final runs
the mean of the two and both values), the difficulty bins over the fixed 0.6B reference (analysis/difficulty_bins.py),
and the MATH-500 ladder by level (graded `knows`). Writes <out>.md, <out>.json and two figures: <out>_benchmarks.png
(per-benchmark saving and accuracy change, one panel each) and <out>_ladder.png (saving and accuracy change by MATH
level). Missing runs are skipped, so the report can be produced while the larger model is still being evaluated.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.difficulty_bins import REFERENCE_BINS, load, summary  # noqa: E402
from evaluation.compare import compare  # noqa: E402

SERIES = {"0.6B": "#2a78d6", "1.7B": "#eb6834"}     # validated categorical slots 1 and 2
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BENCH = ("bfcl", "ifeval", "mbpp", "svamp")


def parse_pairs(items):
    out = {}
    for item in items:
        name, spec = item.split("=", 1)
        base, news = spec.split(":", 1)
        out[name] = (base, [n for n in news.split(",") if n])
    return out


def exists(run):
    return (Path(run) / "records.jsonl").exists()


def main_benchmarks(base, news):
    runs = [compare(base, n) for n in news if exists(n) and (Path(n) / "results.json").exists()]
    if not runs:
        return None
    out = {}
    for b in BENCH:
        vals = [r["benchmarks"][b] for r in runs if b in r["benchmarks"]]
        if vals:
            out[b] = {"saved": [1 - v["token_ratio"] for v in vals], "acc": [v["accuracy_delta"] for v in vals]}
    out["average"] = {"saved": [r["average"]["token_reduction"] for r in runs],
                      "acc": [r["average"]["accuracy_delta"] for r in runs],
                      "saved_ci": [r["average"]["interval"]["token_reduction"] for r in runs],
                      "acc_ci": [r["average"]["interval"]["accuracy_delta"] for r in runs]}
    return out


def bins(base, new, reference):
    if not (exists(base) and exists(new) and exists(reference)):
        return None
    b, n, ref = load(base, "correct"), load(new, "correct"), load(reference, "correct")
    solved = {}
    for (group, qid, _), r in ref.items():
        solved[(group, qid)] = solved.get((group, qid), 0) + bool(r["correct"])
    table = {}
    for group in BENCH:
        keys = [k for k in b if k[0] == group and k in n and b[k]["correct"] is not None and n[k]["correct"] is not None]
        table[group] = {}
        for name, lo, hi in REFERENCE_BINS:
            chosen = [k for k in keys if lo <= solved.get((k[0], k[1]), -1) <= hi]
            if len({k[1] for k in chosen}) >= 2:
                table[group][name] = summary([b[k] for k in chosen], [n[k] for k in chosen])
    return table


def ladder(base, new):
    if not (exists(base) and exists(new)):
        return None
    b, n = load(base, "knows"), load(new, "knows")
    keys = [k for k in b if k in n and k[0] == "math500"]
    out = {}
    for level in range(1, 6):
        chosen = [k for k in keys if b[k]["level"] == level]
        if chosen:
            out[level] = summary([b[k] for k in chosen], [n[k] for k in chosen])
    if keys:
        out["all"] = summary([b[k] for k in keys], [n[k] for k in keys])
    return out


def mean(values):
    return sum(values) / len(values)


def markdown(report):
    lines = ["# Scaling report", "", "## Main benchmarks (mean of the final full runs; individual runs in brackets)", "",
             "| size | " + " | ".join(BENCH) + " | average |", "|---|" + "---|" * (len(BENCH) + 1)]
    for size, data in report["main"].items():
        if not data:
            continue
        cells = []
        for b in BENCH + ("average",):
            if b in data:
                s, a = data[b]["saved"], data[b]["acc"]
                cells.append(f"{100 * mean(s):+.1f}% / {100 * mean(a):+.2f} pp"
                             + (f" ({', '.join(f'{100 * x:+.1f}%/{100 * y:+.2f}' for x, y in zip(s, a))})" if len(s) > 1 else ""))
            else:
                cells.append("–")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines += ["", "## Difficulty bins (0.6B-base solve rate, independent reference run): Δacc pp / tokens saved", ""]
    for size, table in report["bins"].items():
        if not table:
            continue
        lines += [f"**{size}**", "", "| benchmark | " + " | ".join(n for n, _, _ in REFERENCE_BINS) + " |",
                  "|---|" + "---|" * len(REFERENCE_BINS)]
        for group, row in table.items():
            lines.append(f"| {group} | " + " | ".join(
                f"{100 * row[n]['acc_delta']:+.2f} / {100 * row[n]['tokens_saved']:+.1f}%" if n in row else "–"
                for n, _, _ in REFERENCE_BINS) + " |")
        lines.append("")
    lines += ["## MATH-500 ladder (knows): base accuracy, Δacc pp, tokens saved", "",
              "| size | " + " | ".join(f"level {i}" for i in range(1, 6)) + " | all |", "|---|" + "---|" * 6]
    for size, table in report["ladder"].items():
        if table:
            lines.append(f"| {size} | " + " | ".join(
                f"{100 * table[k]['base_acc']:.0f}%, {100 * table[k]['acc_delta']:+.1f}, {100 * table[k]['tokens_saved']:+.0f}%"
                if k in table else "–" for k in (1, 2, 3, 4, 5, "all")) + " |")
    return "\n".join(lines) + "\n"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.axhline(0, color=MUTED, linewidth=0.8)


def figures(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = [s for s, d in report["main"].items() if d]
    if sizes:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
        names = BENCH + ("average",)
        width = 0.8 / len(sizes)
        for panel, (key, title) in enumerate((("saved", "Tokens saved (%)"), ("acc", "Accuracy change (pp)"))):
            ax = axes[panel]
            style(ax)
            for i, size in enumerate(sizes):
                xs = [j + (i - (len(sizes) - 1) / 2) * width for j in range(len(names))]
                ys = [100 * mean(report["main"][size][b][key]) if b in report["main"][size] else 0 for b in names]
                ax.bar(xs, ys, width=width - 0.04, color=SERIES.get(size, MUTED), label=size, zorder=2)
                for x, y in zip(xs, ys):
                    ax.annotate(f"{y:+.1f}", (x, y), xytext=(0, 3 if y >= 0 else -3), textcoords="offset points",
                                ha="center", va="bottom" if y >= 0 else "top", fontsize=7, color=INK)
            ax.margins(y=0.12)
            ax.set_xticks(range(len(names)), [n.upper() if n != "average" else "Average" for n in names])
            ax.set_title(title, loc="left", fontsize=10, color=INK)
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle("Same recipe at two model sizes: main benchmarks", x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout()
        fig.savefig(f"{out}_benchmarks.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)
    ladders = {s: t for s, t in report["ladder"].items() if t}
    if ladders:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
        for panel, (key, title) in enumerate((("tokens_saved", "Tokens saved (%)"),
                                              ("acc_delta", "Accuracy change, lenient (pp)"))):
            ax = axes[panel]
            style(ax)
            for size, table in ladders.items():
                levels = [k for k in range(1, 6) if k in table]
                ys = [100 * table[k][key] for k in levels]
                ax.plot(levels, ys, color=SERIES.get(size, MUTED), linewidth=2, marker="o", markersize=6, zorder=3)
                ax.annotate(size, (levels[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            color=INK, fontsize=9, va="center")
            ax.set_xticks(range(1, 6), [f"Level {k}" for k in range(1, 6)])
            ax.set_xlim(0.7, 5.6)
            ax.set_title(title, loc="left", fontsize=10, color=INK)
        fig.suptitle("MATH-500 held-out test problems: saving and accuracy by difficulty", x=0.01, ha="left",
                     fontsize=11, color=INK)
        fig.tight_layout()
        fig.savefig(f"{out}_ladder.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", action="append", default=[], help='"name=base_run:final_run[,final_run2]"')
    parser.add_argument("--ladder", action="append", default=[], help='"name=base_heldout:new_heldout"')
    parser.add_argument("--reference", required=True, help="Independent 0.6B base run defining the bins")
    arguments = parser.parse_args()
    sizes, ladders = parse_pairs(arguments.size), parse_pairs(arguments.ladder)
    report = {"main": {s: main_benchmarks(b, n) for s, (b, n) in sizes.items()},
              "bins": {s: bins(b, n[0], arguments.reference) if n else None for s, (b, n) in sizes.items()},
              "ladder": {s: ladder(b, n[0]) for s, (b, n) in ladders.items()}}
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{arguments.out}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    text = markdown(report)
    Path(f"{arguments.out}.md").write_text(text, encoding="utf-8")
    print(text)
    figures(report, arguments.out)


if __name__ == "__main__":
    main()
