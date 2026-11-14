#!/usr/bin/env bash
# Step 2 for Qwen3-4B: everything that only generates. No training.
#   1. rollouts   the base model's own answers on the four training pools (1 500-1 600 questions x 4 each)
#   2. base       the base model on the four evaluation benchmarks (8 samples, full protocol)
#   3. heldout    the base model on the held-out sets (HumanEval, GSM8K test, MATH-500)
#
#     bash chains/setup.sh                                            # checks the environment and the pool files
#     nohup bash chains/generate_4b.sh > generate_4b.log 2>&1 &       # or inside tmux
#     tail -f generate_4b.log
#
# Safe to stop and start again: every part continues where it stopped, finished parts are skipped.
# Optional environment:
#     BASE_MODEL=Qwen/Qwen3-4B         the model (default)
#     STAGES="rollouts base heldout"   which parts to run (default: all three, in this order)
#     POOLS="math code tool constraint"   which pools the rollout part covers
#   For a quick plumbing test only: QUESTIONS=8 (questions per pool), EVAL_ARGS="--limit 3 --samples 2",
#   HELDOUT_ARGS="--sets gsm8k_test".
#   Two cards: run the rollouts on one and the two evaluations on the other, each with its own log:
#     CUDA_VISIBLE_DEVICES=0 STAGES=rollouts nohup bash chains/generate_4b.sh > generate_a.log 2>&1 &
#     CUDA_VISIBLE_DEVICES=1 STAGES="base heldout" nohup bash chains/generate_4b.sh > generate_b.log 2>&1 &
# When a run ends it packs results_4b_step2.tar.gz (everything under runs/4b plus the logs) - send that back.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate || exit 1
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B} PYTHONUNBUFFERED=1
STAGES=${STAGES:-rollouts base heldout}
POOLS=${POOLS:-math code tool constraint}
OUT=runs/4b
mkdir -p $OUT
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
minutes() { echo $(( ($(date +%s) - $1) / 60 )); }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }

stamp "step 2 start: model $BASE_MODEL, stages: $STAGES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse --short HEAD 2>/dev/null | sed 's/^/code at commit /'

# pool  sources  questions  generation cap
ROLLOUTS="
math       gsm8k,math                  1500 4096
code       kodcode                     1500 6144
tool       glaive_wide,glaive_removed  1500 4096
constraint dolly_constraints           1600 4096
"
if has rollouts; then
    echo "$ROLLOUTS" | while read -r pool sources questions cap; do
        [ -z "$pool" ] && continue
        questions=${QUESTIONS:-$questions}
        case " $POOLS " in *" $pool "*) ;; *) continue;; esac
        [ -s data/pool4b_$pool.jsonl ] || { stamp "rollouts $pool: data/pool4b_$pool.jsonl is missing - unpack the data archive"; continue; }
        stamp "rollouts $pool start ($questions questions x 4, cap $cap)"
        start=$(date +%s)
        python stopping/generate_rollouts.py --out $OUT/rollouts_$pool --data data/pool4b_$pool.jsonl --sources $sources \
            --questions $questions --max-tokens $cap >> $OUT/rollouts_$pool.log 2>&1 < /dev/null
        status=$?
        grep -a "questions, \|questions done" $OUT/rollouts_$pool.log | tail -1
        rows=$(wc -l < $OUT/rollouts_$pool/rollouts.jsonl 2>/dev/null || echo 0)
        if [ $status -eq 0 ] && [ "$rows" -eq $((questions * 4)) ]; then
            stamp "rollouts $pool done in $(minutes $start) min ($rows rollouts)"
        else
            stamp "rollouts $pool INCOMPLETE (exit $status, $rows of $((questions * 4)) rollouts) - last lines of $OUT/rollouts_$pool.log:"
            tail -5 $OUT/rollouts_$pool.log
        fi
    done
fi

if has base; then
    if [ -s $OUT/base/results.json ]; then
        stamp "base: already finished, skipped"
    else
        if [ ! -s bench/questions.jsonl ]; then
            stamp "benchmarks: download (about 2 MB, pinned sources)"
            python -m evaluation.sources --bench bench >> $OUT/bench.log 2>&1 || { stamp "benchmark download FAILED:"; tail -5 $OUT/bench.log; }
        fi
        if [ -s bench/questions.jsonl ]; then
            stamp "base start (4 benchmarks, 8 samples; MBPP+ comes first and is the slowest)"
            start=$(date +%s)
            python -m evaluation.run --out $OUT/base --model "$BASE_MODEL" $EVAL_ARGS >> $OUT/base.log 2>&1 < /dev/null
            if [ -s $OUT/base/results.json ]; then
                stamp "base done in $(minutes $start) min"
            else
                stamp "base INCOMPLETE - last lines of $OUT/base.log:"; tail -8 $OUT/base.log
            fi
        fi
    fi
fi

if has heldout; then
    if [ -s $OUT/heldout_base/records.jsonl ]; then
        stamp "heldout: already finished, skipped"
    else
        stamp "heldout start (HumanEval, GSM8K test, MATH-500)"
        start=$(date +%s)
        python analysis/heldout_eval.py --out $OUT/heldout_base $HELDOUT_ARGS >> $OUT/heldout_base.log 2>&1 < /dev/null
        if [ -s $OUT/heldout_base/records.jsonl ]; then
            stamp "heldout done in $(minutes $start) min"
        else
            stamp "heldout INCOMPLETE - last lines of $OUT/heldout_base.log:"; tail -8 $OUT/heldout_base.log
        fi
    fi
fi

python - <<'PY'
import json
from pathlib import Path
out = Path("runs/4b")
print("\npool        rollouts  knows   strict  capped  mean tokens")
for path in sorted(out.glob("rollouts_*/rollouts.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    n = len(rows)
    print(f"{path.parent.name[9:]:11s} {n:8d}  {sum(r['knows'] for r in rows) / n:6.1%}  {sum(r['correct'] for r in rows) / n:6.1%}  "
          f"{sum(not r['finished'] for r in rows) / n:6.1%}  {sum(r['tokens'] for r in rows) / n:8.0f}")
results = out / "base/results.json"
if results.exists():
    print("\nbase model on the benchmarks:")
    for name, r in json.loads(results.read_text())["results"].items():
        if isinstance(r, dict) and "accuracy" in r:
            print(f"  {name:20s} accuracy {r['accuracy']:.2%}  mean tokens {r.get('mean_output_tokens', 0):.0f}"
                  f"  capped {r.get('truncation_rate', 0):.1%}")
PY
tar czf results_4b_step2.tar.gz $OUT generate_4b.log generate_a.log generate_b.log 2>/dev/null
stamp "ALL DONE (stages: $STAGES) - send back results_4b_step2.tar.gz ($(du -h results_4b_step2.tar.gz | cut -f1))"
