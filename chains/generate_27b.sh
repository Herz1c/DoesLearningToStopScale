#!/usr/bin/env bash
# Step 2 for Qwen3.8-27B: everything that only generates. No training.
#   1. rollouts   the base model's own answers on the four training pools (1 500-1 600 questions x 4 each)
#   2. base       the base model on the four evaluation benchmarks (8 samples, full protocol)
#   3. heldout    the base model on the held-out sets (HumanEval, GSM8K test, MATH-500)
#
#     SPLIT_GPUS="0 1" nohup bash chains/generate_27b.sh >> generate_27b.log 2>&1 &     # two cards (or inside tmux)
#     nohup bash chains/generate_27b.sh >> generate_27b.log 2>&1 &                      # one card
#     tail -f generate_27b.log
#
# Needs: the environment .venv27 that chains/probe_27b.sh built, the four data/pool27b_*.jsonl files
# (qwen27b_pools.tar.gz unpacked in the repository), internet for the benchmark download.
# Safe to stop and start again: every part continues where it stopped, finished parts are skipped.
# Without bubblewrap (bwrap) model-written code cannot be run on this machine. Everything is still generated;
# the code answers (code pool, MBPP+, HumanEval) are graded afterwards, elsewhere, from the result archive,
# and read 0% or "not graded" in the tables printed here.
# Optional environment:
#     BASE_MODEL=Qwen/Qwen3.8-27B      the model (default)
#     SPLIT_GPUS="0 1"                 two cards at once: code and math rollouts on the first; tool and constraint
#                                      rollouts, benchmarks and held-out sets on the second (about equal work)
#     STAGES="rollouts base heldout"   which parts to run (default: all three, in this order)
#     POOLS="math code tool constraint"   which pools the rollout part covers
#     VENV=.venv27                     the Python environment
#   For a quick plumbing test only: QUESTIONS=8 (questions per pool), EVAL_ARGS="--limit 3 --samples 2",
#   HELDOUT_ARGS="--limit 6".
# When it ends it packs results_27b_step2.tar.gz (everything under runs/27b plus the logs) - send that back.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 1
source "${VENV:-.venv27}/bin/activate" || { echo "no environment: run chains/probe_27b.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
STAGES=${STAGES:-rollouts base heldout}
POOLS=${POOLS:-math code tool constraint}
OUT=runs/27b
mkdir -p $OUT
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
minutes() { echo $(( ($(date +%s) - $1) / 60 )); }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
source chains/gpu_wait.sh
[ -z "$CHILD" ] && one_chain $OUT/.generate.lock

# twice <log> <command...>: the command after the card is free; a second try if the first one fails
# (every part continues where it stopped, so a second try loses nothing).
twice() {
    local log=$1 attempt; shift
    for attempt in 1 2; do
        settle
        "$@" >> "$log" 2>&1 < /dev/null && return 0
        [ $attempt -eq 1 ] && stamp "first try failed, trying once more (details in $log)"
    done
    return 1
}

stamp "step 2 start: model $BASE_MODEL, stages: $STAGES, pools: $POOLS${CUDA_VISIBLE_DEVICES:+, GPU $CUDA_VISIBLE_DEVICES}"
if [ -z "$CHILD" ]; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"; git status --short 2>/dev/null | head -5
    for pool in math code tool constraint; do [ -s data/pool27b_$pool.jsonl ] || echo "WARNING: data/pool27b_$pool.jsonl is missing - unpack qwen27b_pools.tar.gz in the repository"; done
    # The graders' small packages (benchmark download, IFEval, BFCL and MBPP+ parsing) into the environment.
    if ! python -c "import datasets, nltk, langdetect, absl, immutabledict, tree_sitter_python, termcolor, appdirs, multipledispatch, tempdir, wget" 2>/dev/null; then
        stamp "installing the graders' packages"
        UV=$(command -v uv || echo "$PWD/.venv_uv/bin/uv")
        "$UV" pip install --python "$VIRTUAL_ENV/bin/python" "datasets>=3" appdirs termcolor wget tempdir multipledispatch \
            tree-sitter tree-sitter-python absl-py langdetect nltk immutabledict > $OUT/install.log 2>&1 \
            || { stamp "install FAILED - last lines of $OUT/install.log:"; tail -5 $OUT/install.log; }
    fi
    if has base && [ ! -s bench/questions.jsonl ]; then
        stamp "benchmarks: download (about 2 MB, pinned sources)"
        python -m evaluation.sources --bench bench >> $OUT/bench.log 2>&1 || { stamp "benchmark download FAILED:"; tail -5 $OUT/bench.log; }
    fi
fi
if bwrap --unshare-all --ro-bind / / true 2>/dev/null; then
    SANDBOX=bwrap
    # bwrap starting is not enough (2026-11-18: it started, yet the training-code grader failed every program)
    [ -z "$CHILD" ] && python -c "import sys; sys.path.insert(0, '.'); from stopping.code_tasks import self_test; e = self_test(); print('code sandbox self-test: ' + (e or 'OK'))"
else
    SANDBOX=none
    [ -z "$CHILD" ] && echo "NOTE: bubblewrap (bwrap) cannot start a sandbox here, so model-written code is not run on this machine." \
        && echo "      Code answers are generated and graded later, elsewhere; here they read 0% or 'not graded'."
fi

# Two cards: SPLIT_GPUS="0 1". Each half is this script again on one card, with its own log.
if [ -n "$SPLIT_GPUS" ]; then
    set -- $SPLIT_GPUS
    for gpu in $1 $2; do
        used=$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -z "$used" ] && { stamp "GPU $gpu does not exist here (nvidia-smi -L lists the cards). Nothing started."; exit 1; }
        [ "$used" -gt "${GPU_FREE_MIB:-8000}" ] && stamp "WARNING: GPU $gpu already has $used MiB in use - the half on it will wait for it"
    done
    stamp "two cards: code + math rollouts on GPU $1 (generate_27b.gpu$1.log); tool + constraint rollouts, benchmarks, held-out on GPU $2 (generate_27b.gpu$2.log)"
    CHILD=1 SPLIT_GPUS= CUDA_VISIBLE_DEVICES=$1 STAGES="rollouts" POOLS="code math" bash "$SELF" > generate_27b.gpu$1.log 2>&1 &
    CHILD=1 SPLIT_GPUS= CUDA_VISIBLE_DEVICES=$2 STAGES="rollouts base heldout" POOLS="tool constraint" bash "$SELF" > generate_27b.gpu$2.log 2>&1 &
    wait
    grep -h -a "done in\|INCOMPLETE\|FAILED\|WARNING\|skipped" generate_27b.gpu$1.log generate_27b.gpu$2.log
else

# pool  sources  questions  generation cap
ROLLOUTS="
code       kodcode            1500 6144
math       math               1500 4096
tool       glaive_removed     1500 4096
constraint dolly_constraints  1600 4096
"
if has rollouts; then
    echo "$ROLLOUTS" | while read -r pool sources questions cap; do
        [ -z "$pool" ] && continue
        questions=${QUESTIONS:-$questions}
        case " $POOLS " in *" $pool "*) ;; *) continue;; esac
        [ -s data/pool27b_$pool.jsonl ] || { stamp "rollouts $pool: data/pool27b_$pool.jsonl is missing - unpack the data archive"; continue; }
        stamp "rollouts $pool start ($questions questions x 4, cap $cap)"
        start=$(date +%s)
        twice $OUT/rollouts_$pool.log python stopping/generate_rollouts.py --out $OUT/rollouts_$pool \
            --data data/pool27b_$pool.jsonl --sources $sources --questions $questions --max-tokens $cap
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
    elif [ ! -s bench/questions.jsonl ]; then
        stamp "base FAILED: no bench/questions.jsonl (the benchmark download failed, see $OUT/bench.log)"
    else
        stamp "base start (4 benchmarks, 8 samples; MBPP+ comes first and is the slowest)"
        start=$(date +%s)
        twice $OUT/base.log python -m evaluation.run --out $OUT/base --model "$BASE_MODEL" --sandbox $SANDBOX $EVAL_ARGS
        if [ -s $OUT/base/results.json ]; then
            stamp "base done in $(minutes $start) min"
        else
            stamp "base INCOMPLETE - last lines of $OUT/base.log:"; tail -8 $OUT/base.log
        fi
    fi
fi

if has heldout; then
    if [ -s $OUT/heldout_base/records.jsonl ]; then
        stamp "heldout: already finished, skipped"
    else
        stamp "heldout start (HumanEval, GSM8K test, MATH-500)"
        start=$(date +%s)
        twice $OUT/heldout_base.log python analysis/heldout_eval.py --out $OUT/heldout_base $HELDOUT_ARGS
        if [ -s $OUT/heldout_base/records.jsonl ]; then
            stamp "heldout done in $(minutes $start) min"
        else
            stamp "heldout INCOMPLETE - last lines of $OUT/heldout_base.log:"; tail -8 $OUT/heldout_base.log
        fi
    fi
fi
fi
[ -n "$CHILD" ] && { stamp "this card's parts finished"; exit 0; }

python - <<'PY'
import json
from pathlib import Path
out = Path("runs/27b")
print("\npool        rollouts  knows   strict  capped  mean tokens")
for path in sorted(out.glob("rollouts_*/rollouts.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    n = len(rows)
    print(f"{path.parent.name[9:]:11s} {n:8d}  {sum(bool(r['knows']) for r in rows) / n:6.1%}  {sum(bool(r['correct']) for r in rows) / n:6.1%}  "
          f"{sum(not r['finished'] for r in rows) / n:6.1%}  {sum(r['tokens'] for r in rows) / n:8.0f}")
results = out / "base/results.json"
if results.exists():
    print("\nbase model on the benchmarks:")
    for name, r in json.loads(results.read_text())["results"].items():
        if isinstance(r, dict) and "accuracy" in r:
            accuracy = "not graded" if r["accuracy"] is None else f"{r['accuracy']:.2%}"
            print(f"  {name:20s} accuracy {accuracy:>10s}  mean tokens {r.get('mean_output_tokens') or 0:.0f}"
                  f"  capped {r.get('truncation_rate') or 0:.1%}  not graded {r.get('ungraded', 0)}")
held = out / "heldout_base/records.jsonl"
if held.exists():
    print("\nbase model on the held-out sets:")
    sets = {}
    for line in held.read_text().splitlines():
        r = json.loads(line)
        sets.setdefault(r["set"], []).append(r)
    for name, rows in sets.items():
        graded = [r for r in rows if r["correct"] is not None]
        accuracy = f"{sum(r['correct'] for r in graded) / len(graded):.2%}" if graded else "not graded"
        print(f"  {name:20s} accuracy {accuracy:>10s}  mean tokens {sum(r['tokens'] for r in rows) / len(rows):.0f}"
              f"  capped {sum(r['truncated'] for r in rows) / len(rows):.1%}")
PY
tar czf results_27b_step2.tar.gz $OUT generate_27b*.log 2>/dev/null || tar czf results_27b_step2.tar.gz $OUT
stamp "ALL DONE (stages: $STAGES) - send back results_27b_step2.tar.gz ($(du -h results_27b_step2.tar.gz | cut -f1))"
