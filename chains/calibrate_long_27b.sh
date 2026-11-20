#!/usr/bin/env bash
# Calibration of the long-reasoning track for Qwen3.8-27B: how long the base model thinks on the new, harder
# training sources, and from where in its thinking it already has the answer. Generation only, no training.
#
#     nohup bash chains/calibrate_long_27b.sh >> calibrate_long_27b.log 2>&1 &        # or inside tmux
#     tail -f calibrate_long_27b.log
#
# Needs: the environment .venv27, the file data/long_cal_all.jsonl (unpack long_cal_pools.tar.gz in the repo
# folder), internet. Two cards (GPUS="0 1", the default): each takes half of the questions, with its own log
# (calibrate_long_27b.gpu<N>.log). With one card (GPUS="0") the halves run one after the other.
# Steps: 1. rollouts (2 per question, temperature 1.0, up to 65 536 tokens)   2. forced answers at up to 12
# points of every rollout   3. first card: GPQA-Diamond for the base model, if the login allows it (accept the
# terms at https://huggingface.co/datasets/Idavidrein/gpqa, then `hf auth login`); second card: how long a
# sequence can be trained (memory probe).
# Safe to stop and start again: finished questions and rollouts are kept.
# Optional environment: GPUS="0 1"; STAGES="rollouts readouts extras"; for a plumbing test only:
#     CAL_ARGS="--limit 12 --max-tokens 1500" PROBE_LENGTHS=2048,4096
# When it ends it packs results_27b_long_cal.tar.gz - send that back.
cd "$(dirname "$0")/.." || exit 1
source "${VENV:-.venv27}/bin/activate" || { echo "no environment: run chains/probe_27b.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
STAGES=${STAGES:-rollouts readouts extras}
set -- ${GPUS:-0 1}
GA=$1 GB=${2:-$1}
R=runs/27b_long_cal
REF=runs/27b_reference
mkdir -p $R
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
source chains/gpu_wait.sh
one_chain $R/.calibrate.lock

stamp "long-reasoning calibration start: model $BASE_MODEL, stages: $STAGES, cards: $GA $GB"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"
[ -s data/long_cal_all.jsonl ] || { stamp "data/long_cal_all.jsonl is missing: unpack long_cal_pools.tar.gz in $(pwd)"; exit 1; }
echo "questions: $(wc -l < data/long_cal_all.jsonl)"
if ! python -c "import math_verify, pandas, pyarrow" 2>/dev/null; then
    stamp "installing the graders' packages"
    UV=$(command -v uv || echo "$PWD/.venv_uv/bin/uv")
    "$UV" pip install --python "$VIRTUAL_ENV/bin/python" "math-verify[antlr4_13_2]" emoji syllapy pandas pyarrow \
        absl-py langdetect nltk immutabledict > $R/install.log 2>&1 || { stamp "install FAILED - last lines of $R/install.log:"; tail -5 $R/install.log; }
fi
sandbox=$(python -c "import sys; sys.path.insert(0, '.'); from stopping.code_tasks import self_test; print(self_test() or 'OK')" 2>&1 | tail -1)
echo "code sandbox self-test: $sandbox   (if not OK, the programs are graded later, elsewhere)"
python stopping/long_tasks.py --judge-timing 2>&1 | grep -a "judge timing"

# job <name> <command...>: one step on the card in CUDA_VISIBLE_DEVICES; two tries; the step's own log
job() {
    local name=$1 attempt start; shift
    stamp "$name start (GPU $CUDA_VISIBLE_DEVICES)"
    start=$(date +%s)
    for attempt in 1 2; do
        settle
        "$@" >> $R/$name.log 2>&1 < /dev/null && { stamp "$name done in $(( ($(date +%s) - start) / 60 )) min"; return 0; }
        [ $attempt -eq 1 ] && stamp "$name: first try failed, trying once more (details in $R/$name.log)"
    done
    stamp "$name FAILED - last lines of $R/$name.log:"; tail -8 $R/$name.log; return 1
}
half() {     # half <index>: this card's half of the questions, then of the forced answers
    if has rollouts; then job rollouts$1 python stopping/long_rollouts.py --data data/long_cal_all.jsonl --out $R --shard $1/2 $CAL_ARGS; fi
    if has readouts; then job readouts$1 python stopping/readouts.py $R --shard $1/2; fi
}
gpqa() {
    python -m evaluation.reference --prepare 2>&1 | grep -a -v -i "warn" > $R/prepare.log
    if grep -q "NOT AVAILABLE: gpqa" $R/prepare.log || ! grep -q "gpqa_diamond" $R/prepare.log; then
        echo "NOTE: GPQA-Diamond is left out (no access). Accept its terms, log in (hf auth login), start this chain again."
    elif [ ! -s $REF/base/records.jsonl ]; then
        echo "NOTE: $REF/base is not here (the earlier reference run) - GPQA-Diamond is left for later."
    else
        job gpqa_base python -m evaluation.reference --out $REF/base --model "$BASE_MODEL" --samples 4 --sets gpqa_diamond --no-code
    fi
}
probe() { job memory_probe python stopping/memory_probe.py ${PROBE_LENGTHS:+--lengths $PROBE_LENGTHS}; }

if [ "$GA" != "$GB" ]; then
    ( export CUDA_VISIBLE_DEVICES=$GA; half 0; if has extras; then gpqa; fi ) > calibrate_long_27b.gpu$GA.log 2>&1 &
    ( export CUDA_VISIBLE_DEVICES=$GB; half 1; if has extras; then probe; fi ) > calibrate_long_27b.gpu$GB.log 2>&1 &
    wait
    cat calibrate_long_27b.gpu$GA.log calibrate_long_27b.gpu$GB.log
else
    ( export CUDA_VISIBLE_DEVICES=$GA; half 0; half 1; if has extras; then gpqa; probe; fi )
fi

echo; echo "== rollouts by part (knows = right answer however formatted; capped = ran into the token limit)"
python - <<'PY' 2>/dev/null | tee runs/27b_long_cal/rollouts_summary.txt
import collections, glob, json, statistics as st
rows = [json.loads(l) for f in sorted(glob.glob("runs/27b_long_cal/rollouts*.jsonl")) for l in open(f, encoding="utf-8") if l.strip().endswith("}")]
parts = collections.defaultdict(list)
for r in rows: parts[r.get("part", "-")].append(r)
print(f"{'part':16} {'rollouts':>8} {'knows':>6} {'strict':>6} {'capped':>6} {'mean think':>10} {'median':>7} {'p90':>7} {'tokens M':>8}")
for part, g in sorted(parts.items()):
    t = sorted(r["thinking_tokens"] for r in g)
    print(f"{part:16} {len(g):8} {100*st.mean(r['knows'] for r in g):5.1f}% {100*st.mean(r['correct'] for r in g):5.1f}% "
          f"{100*st.mean(not r['finished'] for r in g):5.1f}% {st.mean(t):10.0f} {t[len(t)//2]:7} {t[9*len(t)//10]:7} {sum(r['tokens'] for r in g)/1e6:8.2f}")
print(f"total {len(rows)} rollouts, {sum(r['tokens'] for r in rows)/1e6:.1f} M tokens")
PY
echo; echo "== from where the answer is settled"
python stopping/readouts.py $R --report 2>/dev/null | tee $R/readouts_report.txt
echo; echo "== memory probe"; grep -a -A8 "^model \|tokens .*forward" $R/memory_probe.log 2>/dev/null | grep -a -v -i "warn" | tail -9
tar czf results_27b_long_cal.tar.gz $R $REF/base calibrate_long_27b*.log 2>/dev/null
stamp "ALL DONE - send back results_27b_long_cal.tar.gz ($(du -h results_27b_long_cal.tar.gz | cut -f1))"
