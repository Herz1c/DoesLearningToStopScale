#!/usr/bin/env bash
# (Also runs the same pilot on Qwen3-0.6B: BASE_MODEL=Qwen/Qwen3-0.6B TAG=06b, see notebooks/long_06b_colab.ipynb.)
# The long-reasoning pilot repeated on Qwen3-4B: the same questions, method and rule as on Qwen3.8-27B, to see
# whether the smaller model shows the same failure (an adapter that learns a typical length instead of when its
# answer is settled). If it does, 4B is the test bed for the next method. Registered in PREREGISTRATION.md.
#   1. calibration  the base model on the 962 calibration questions (2 rollouts each), then answers forced at
#                   up to 12 points of every rollout                                   -> runs/4b_long_cal
#   2. pilot        chains/long_27b.sh, stage "pilot": validation questions for the base model, training rows,
#                   both ways of training, both on the validation questions, the rule  -> runs/4b_long
#
#     nohup bash chains/long_4b.sh >> long_4b.log 2>&1 &        # or inside tmux
#     tail -f long_4b.log
#
# Needs: the environment .venv (chains/setup.sh); data/long_cal_all.jsonl (long_cal_pools.tar.gz) and
# data/long_val_all.jsonl, data/long_full_all.jsonl (long_pools.tar.gz), unpacked in the repo folder.
# One card is enough (GPUS="0", the default here); with GPUS="0 1" the work is shared.
# Safe to stop and start again with the same command: finished parts are skipped.
# Generation is capped at 32 768 tokens: the model's context is 40 960.
# Optional environment: GPUS="0"; JUDGE_WORKERS=2; BASE_MODEL with TAG (Qwen/Qwen3-0.6B, TAG=06b); VENV=none;
# NO_LOCK=1 (file systems without flock); SANDBOX=user (see stopping/code_tasks.py); for a plumbing test only: MAX_TOKENS, CAL_DATA, VAL_DATA,
# TRAIN_ARGS, CAL_ARGS.
# At the end (or at a stop) it packs results_4b_long.tar.gz - send that back.
cd "$(dirname "$0")/.." || exit 1
export VENV=${VENV:-.venv}        # VENV=none: the Python already on the PATH (a Colab runtime)
[ "$VENV" = none ] || source "$VENV/bin/activate" || { echo "no environment: run chains/setup.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
TAG=${TAG:-4b}                     # names of the folders, the log and the archive: TAG=06b for Qwen3-0.6B
export JUDGE_WORKERS=${JUDGE_WORKERS:-2} GPUS=${GPUS:-0}
MAX_TOKENS=${MAX_TOKENS:-32768}
CAL=${CAL:-runs/${TAG}_long_cal}
CAL_DATA=${CAL_DATA:-data/long_cal_all.jsonl}
set -- $GPUS
GA=$1
mkdir -p $CAL
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
source chains/gpu_wait.sh
one_chain $CAL/.long4b.lock

stamp "long reasoning on $BASE_MODEL: calibration, then the pilot; cards: $GPUS, cap $MAX_TOKENS tokens"
[ -s $CAL_DATA ] || { stamp "$CAL_DATA is missing: unpack long_cal_pools.tar.gz in $(pwd)"; exit 1; }
if ! python -c "import math_verify, pandas, pyarrow" 2>/dev/null; then
    stamp "installing the graders' packages"
    python -m pip install --quiet "math-verify[antlr4_13_2]" emoji syllapy pandas pyarrow absl-py langdetect nltk immutabledict \
        > $CAL/install.log 2>&1 || { stamp "install FAILED - last lines of $CAL/install.log:"; tail -5 $CAL/install.log; exit 1; }
fi

# part <label> <marker> <log> <command...>: one step on the first card, skipped when its marker exists; two tries
part() {
    local label=$1 marker=$2 log=$3 attempt start; shift 3
    if [ -s "$marker" ]; then stamp "$label: already done, skipped"; return 0; fi
    stamp "$label start"
    start=$(date +%s)
    for attempt in 1 2; do
        "$@" >> "$log" 2>&1 < /dev/null && { date > "$marker"; stamp "$label done in $(( ($(date +%s) - start) / 60 )) min"; return 0; }
        [ $attempt -eq 1 ] && stamp "$label: first try failed, trying once more (details in $log)"
    done
    stamp "$label FAILED - last lines of $log:"; tail -8 "$log"
    tar czf results_${TAG}_long.tar.gz --exclude="rollouts*.jsonl*" $CAL long_${TAG}*.log 2>/dev/null
    stamp "STOPPED at: $label - send back results_${TAG}_long.tar.gz and long_${TAG}.log, then start the same command again after the fix"
    exit 1
}
engine() { ( export CUDA_VISIBLE_DEVICES=$GA; settle; "$@" ); }

part "calibration rollouts" $CAL/.rollouts $CAL/rollouts.log \
    engine python stopping/long_rollouts.py --data $CAL_DATA --out $CAL --max-tokens $MAX_TOKENS --wave 2000 --judge-later $CAL_ARGS
part "calibration: programs judged" $CAL/.rollouts_judged $CAL/judge.log python stopping/readouts.py $CAL --regrade
part "calibration: forced answers" $CAL/.readouts $CAL/readouts.log engine python stopping/readouts.py $CAL --judge-later
part "calibration: forced answers judged" $CAL/.readouts_judged $CAL/judge.log python stopping/readouts.py $CAL --regrade
python stopping/readouts.py $CAL --report 2>/dev/null | tee $CAL/readouts_report.txt | tail -9

stamp "pilot (chains/long_27b.sh, stage pilot)"
CAL=$CAL RUN_DIR=${RUN_DIR:-runs/${TAG}_long} CKPT_DIR=${CKPT_DIR:-checkpoints/${TAG}_long} STAGES=pilot \
    ROLL_ARGS="--max-tokens $MAX_TOKENS" PACK=results_${TAG}_long PACK_EXTRA="$CAL long_${TAG}*.log" bash chains/long_27b.sh
