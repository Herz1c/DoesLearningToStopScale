#!/usr/bin/env bash
# Version 2 (long reasoning) for Qwen3.8-27B, everything up to the RL stage, in one chain:
#   pilot  validation rollouts of the base model -> training rows from the calibration data -> two ways of
#          training, one per card ("trace" = thinking cut where the answer is settled; "decision" = only the
#          decision to end the thinking, held to the base model elsewhere) -> both on the validation questions
#          -> a fixed rule picks the winner. If neither passes the rule the chain STOPS here: that is a result.
#   full   the base model on the new training questions (rollouts, forced answers), both cards
#   main   training with the winning way on calibration + new data, then the validation questions again
#   final  ONLY if the main adapter passes the rule: the long-reasoning reference sets (first card) and the
#          short benchmarks twice + held-out sets (second card)
#
#     nohup bash chains/long_27b.sh >> long_27b.log 2>&1 &        # or inside tmux
#     tail -f long_27b.log
#
# Needs: the environment .venv27; runs/27b_long_cal (rollouts and the repeated code readouts: the folder of the
# two calibration runs); data/long_val_all.jsonl and data/long_full_all.jsonl (unpack long_pools.tar.gz in the
# repo folder); for the final stage runs/27b_reference/base, runs/27b/base, runs/27b/heldout_base (earlier runs)
# and internet. Two cards (GPUS="0 1").
# Safe to stop and start again with the same command: finished parts are skipped, generation continues by
# question, training continues from its last save (every 20 steps).
# Optional environment: GPUS="0 1"; STAGES="pilot full main final"; JUDGE_WORKERS=2 (parallel code judges).
#   For a plumbing test only: ROLL_ARGS, TRAIN_ARGS, REF_ARGS, EVAL_ARGS, HELDOUT_ARGS, CAL=<dir>, VAL_SAMPLES.
# At the end (or at a stop) it packs results_27b_long.tar.gz - send that back.
cd "$(dirname "$0")/.." || exit 1
source "${VENV:-.venv27}/bin/activate" || { echo "no environment: run chains/probe_27b.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
export JUDGE_WORKERS=${JUDGE_WORKERS:-2}
STAGES=${STAGES:-pilot full main final}
set -- ${GPUS:-0 1}
GA=$1 GB=${2:-$1}
[ "$GA" != "$GB" ] && N=2 || N=1
CAL=${CAL:-runs/27b_long_cal}
R=${RUN_DIR:-runs/27b_long}
C=${CKPT_DIR:-checkpoints/27b_long}
V=$R/val
VAL_DATA=${VAL_DATA:-data/long_val_all.jsonl}
FULL_DATA=${FULL_DATA:-data/long_full_all.jsonl}
VAL_SAMPLES=${VAL_SAMPLES:-4}
mkdir -p $R $C $V
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
source chains/gpu_wait.sh

# step <label> <file that proves the part is done> <log> <command...>   (runs on the card in CUDA_VISIBLE_DEVICES)
step() {
    local label=$1 proof=$2 log=$3; shift 3
    if [ -s "$proof" ]; then stamp "$label: already done, skipped"; return 0; fi
    stamp "$label start${CUDA_VISIBLE_DEVICES:+ (GPU $CUDA_VISIBLE_DEVICES)}"
    local start attempt; start=$(date +%s)
    for attempt in 1 2; do      # a second try continues where the first stopped
        [ -n "$CUDA_VISIBLE_DEVICES" ] && settle
        "$@" >> "$log" 2>&1 < /dev/null
        [ -s "$proof" ] && break
        [ $attempt -eq 1 ] && stamp "$label: first try failed, trying once more (details in $log)"
    done
    if [ -s "$proof" ]; then
        stamp "$label done in $(( ($(date +%s) - start) / 60 )) min"
    else
        stamp "$label FAILED - last lines of $log:"; tail -8 "$log"; return 1
    fi
}
# marked <marker> <command...>: the command, and the marker file only when it succeeded
marked() { local marker=$1; shift; "$@" && date > "$marker"; }

finish() {
    for f in $R/pilot.txt $R/main.txt; do [ -s $f ] && { echo; echo "== $f"; grep -a "^overall\|PASS\|^winner\|^==" $f; }; done
    for f in runs/27b_reference/long_sft.compare.txt runs/27b/long_sft_full1.compare.txt runs/27b/long_sft_full2.compare.txt; do
        [ -s $f ] && { echo; echo "== $f"; tail -14 $f; }
    done
    tar czf results_27b_long.tar.gz --exclude="decision.jsonl" --exclude="*.before" --exclude="training_state.json" \
        $R $C runs/27b_reference/long_sft runs/27b_reference/long_sft.* runs/27b/long_sft_full[12]* runs/27b/heldout_long_sft* \
        long_27b*.log 2>/dev/null
    stamp "$1 - send back results_27b_long.tar.gz ($(du -h results_27b_long.tar.gz | cut -f1))"
    exit "${2:-0}"
}
stop() { finish "STOPPED at: $1 (fix or report it, then start the same command again)" 1; }

# lanes <A> <B>: lane_<A> on the first card and lane_<B> on the second, side by side; 1 if either failed
lanes() {
    if [ $N -eq 2 ]; then
        ( export CUDA_VISIBLE_DEVICES=$GA; "lane_$1" ) >> long_27b.gpu$GA.log 2>&1 &
        local a=$!
        ( export CUDA_VISIBLE_DEVICES=$GB; "lane_$2" ) >> long_27b.gpu$GB.log 2>&1 &
        local b=$!
        wait $a; local sa=$?
        wait $b; local sb=$?
        return $(( sa || sb ))
    fi
    ( export CUDA_VISIBLE_DEVICES=$GA; "lane_$1" && "lane_$2" )
}
show() { grep -h -a "^\[" long_27b.gpu$GA.log long_27b.gpu$GB.log 2>/dev/null | sort | grep -a -v " start" | tail -${1:-8}; }
# shares <label> <dir> <log prefix> <command with %K for the share index>: the command on both cards, a share each
shares() {
    local label=$1 dir=$2 log=$3; shift 3
    local template="$*"
    lane_s0() { step "$label 0/$N" $dir/.share0of$N.done $log.0.log marked $dir/.share0of$N.done ${template//%K/0}; }
    lane_s1() { [ $N -eq 1 ] || step "$label 1/$N" $dir/.share1of$N.done $log.1.log marked $dir/.share1of$N.done ${template//%K/1}; }
    mkdir -p $dir
    lanes s0 s1
}
# rollouts_of <run dir> <data> <samples> [--adapter dir]: rollouts on both cards, programs judged afterwards
rollouts_of() {
    local dir=$1 data=$2 samples=$3; shift 3
    shares "rollouts $dir" $dir $dir/rollouts python stopping/long_rollouts.py --data $data --out $dir \
        --samples $samples --shard %K/$N --wave 2000 --judge-later $ROLL_ARGS "$@" || return 1
    step "programs of $dir judged" $dir/.judged $dir/judge.log marked $dir/.judged python stopping/readouts.py $dir --regrade
}
validate() {    # validate <report name> <name=dir ...>
    local name=$1; shift
    python stopping/long_validate.py $V/base "$@" --out $R/$name.json 2>&1 | grep -a -v -i "warn" | tee $R/$name.txt
}

[ -z "$CHILD" ] && one_chain $R/.long.lock
stamp "long reasoning, version 2: model $BASE_MODEL, stages: $STAGES, cards: $GA $GB, judge workers $JUDGE_WORKERS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"
echo "cores: node $(nproc --all), this job $(nproc)"
ls $CAL/rollouts*.jsonl > /dev/null 2>&1 || stop "$CAL/rollouts*.jsonl are missing (the calibration run's folder)"
ls $CAL/readouts*.jsonl > /dev/null 2>&1 || stop "$CAL/readouts*.jsonl are missing (the calibration run's folder)"
[ -s $VAL_DATA ] || stop "$VAL_DATA is missing: unpack long_pools.tar.gz in $(pwd)"
[ -s $FULL_DATA ] || stop "$FULL_DATA is missing: unpack long_pools.tar.gz in $(pwd)"
sandbox=$(python -c "import sys; sys.path.insert(0, '.'); from stopping.code_tasks import self_test; print(self_test() or 'OK')" 2>&1 | tail -1)
echo "code sandbox self-test: $sandbox"
python stopping/long_tasks.py --judge-timing 2>&1 | grep -a "judge timing" | tee $R/judge_timing.txt
grep -a "judge timing" $R/judge_timing.txt | tail -1 | grep -q "24/24 passed" || stop "the code judge fails trivial programs with $JUDGE_WORKERS workers (try JUDGE_WORKERS=1)"

train() {   # train <rows file> <adapter dir>
    step "training $2" $2/training_args.json $2.log python stopping/train_long.py --data $1 --out $2 $TRAIN_ARGS
}

if has pilot; then
    stamp "pilot: the base model on the validation questions"
    rollouts_of $V/base $VAL_DATA $VAL_SAMPLES || stop "validation rollouts of the base model"
    ( export CUDA_VISIBLE_DEVICES=$GA
      step "pilot rows: answers at the settled points" $R/pilot/.sampled $R/pilot_rows.log \
          marked $R/pilot/.sampled python stopping/long_examples.py $CAL --out $R/pilot --sample ) || stop "pilot rows (sampling)"
    step "pilot rows: build" $R/pilot/decision.jsonl $R/pilot_rows.log python stopping/long_examples.py $CAL --out $R/pilot --build \
        || stop "pilot rows (build)"
    grep -a "rollouts$\|^  \|^trace\|^decision" $R/pilot_rows.log | tail -12
    lane_trace() { train $R/pilot/trace.jsonl $C/pilot_trace && step "validation rollouts trace" $V/trace/.done $V/trace.log \
        marked $V/trace/.done python stopping/long_rollouts.py --data $VAL_DATA --out $V/trace --samples $VAL_SAMPLES --wave 2000 --judge-later $ROLL_ARGS --adapter $C/pilot_trace; }
    lane_decision() { train $R/pilot/decision.jsonl $C/pilot_decision && step "validation rollouts decision" $V/decision/.done $V/decision.log \
        marked $V/decision/.done python stopping/long_rollouts.py --data $VAL_DATA --out $V/decision --samples $VAL_SAMPLES --wave 2000 --judge-later $ROLL_ARGS --adapter $C/pilot_decision; }
    mkdir -p $V/trace $V/decision
    stamp "pilot: GPU $GA trains 'trace', GPU $GB trains 'decision'; each then answers the validation questions"
    lanes trace decision; status=$?
    show 8
    [ $status -eq 0 ] || stop "pilot training or its validation rollouts (see long_27b.gpu$GA.log, long_27b.gpu$GB.log)"
    for name in trace decision; do
        step "programs of $V/$name judged" $V/$name/.judged $V/$name/judge.log marked $V/$name/.judged python stopping/readouts.py $V/$name --regrade \
            || stop "judging the programs of $name"
    done
    validate pilot trace=$V/trace decision=$V/decision
fi
[ -s $R/pilot.json ] || stop "the pilot has no result yet ($R/pilot.json): run the pilot stage first"
WINNER=$(python -c "import json; print(json.load(open('$R/pilot.json'))['winner'])")
stamp "pilot result: winner = $WINNER"
[ "$WINNER" = "none" ] && finish "STOPPED BY THE RULE: neither way of training passes on the validation questions (this is a result, not an error)"

if has full; then
    stamp "full data: the base model on the new training questions"
    rollouts_of $R/full $FULL_DATA 2 || stop "rollouts of the full pool"
    shares "forced answers of $R/full" $R/full/readouts_shares $R/full/readouts python stopping/readouts.py $R/full --shard %K/$N --judge-later \
        || stop "forced answers of the full pool"
    step "forced answers judged" $R/full/.readouts_judged $R/full/judge.log marked $R/full/.readouts_judged python stopping/readouts.py $R/full --regrade \
        || stop "judging the forced answers"
    python stopping/readouts.py $R/full --report 2>/dev/null | tee $R/full_readouts_report.txt | tail -8
fi

if has main; then
    [ -s $R/full/.readouts_judged ] || stop "the full data are not ready ($R/full): run the full stage first"
    ( export CUDA_VISIBLE_DEVICES=$GA
      step "main rows: answers at the settled points" $R/main/.sampled $R/main_rows.log \
          marked $R/main/.sampled python stopping/long_examples.py $CAL $R/full --out $R/main --sample ) || stop "main rows (sampling)"
    step "main rows: build" $R/main/decision.jsonl $R/main_rows.log python stopping/long_examples.py $CAL $R/full --out $R/main --build \
        || stop "main rows (build)"
    grep -a "rollouts$\|^  \|^trace\|^decision" $R/main_rows.log | tail -12
    ( export CUDA_VISIBLE_DEVICES=$GA; train $R/main/$WINNER.jsonl $C/main ) || stop "main training"
    rollouts_of $V/main $VAL_DATA $VAL_SAMPLES --adapter $C/main || stop "validation rollouts of the main adapter"
    validate main main=$V/main
fi

if has final; then
    [ -s $R/main.json ] || stop "the main adapter has no validation result yet ($R/main.json)"
    PASSED=$(python -c "import json; print(json.load(open('$R/main.json'))['winner'])")
    [ "$PASSED" = "main" ] || finish "STOPPED BY THE RULE: the main adapter does not pass on the validation questions; the reference sets stay closed (a result, not an error)"
    [ -s runs/27b_reference/base/results.json ] || stop "runs/27b_reference/base is missing (the earlier reference run)"
    [ -s runs/27b/base/results.json ] || stop "runs/27b/base is missing (the short benchmarks of the base model)"
    python -m evaluation.reference --prepare 2>&1 | grep -a -v -i "warn" > $R/prepare.log
    lane_reference() {
        step "reference sets" runs/27b_reference/long_sft/results.json runs/27b_reference/long_sft.log \
            python -m evaluation.reference --out runs/27b_reference/long_sft --samples 4 --model "$BASE_MODEL" --adapter $C/main --workers $JUDGE_WORKERS $REF_ARGS || return 1
        python -m evaluation.reference --compare runs/27b_reference/base runs/27b_reference/long_sft 2>&1 | grep -a -v -i "warn" > runs/27b_reference/long_sft.compare.txt
    }
    short() {    # short <run name>
        step "short benchmarks $1" runs/27b/$1/results.json runs/27b/$1.log \
            python -m evaluation.run --out runs/27b/$1 --model "$BASE_MODEL" --adapter $C/main --workers $JUDGE_WORKERS $EVAL_ARGS || return 1
        python -m evaluation.compare runs/27b/base runs/27b/$1 --json runs/27b/$1.compare.json > runs/27b/$1.compare.txt 2>&1
    }
    lane_short() {
        short long_sft_full1 || return 1
        short long_sft_full2 || return 1
        if [ -s runs/27b/heldout_base/records.jsonl ]; then
            step "held-out" runs/27b/heldout_long_sft/records.jsonl runs/27b/heldout_long_sft.log \
                python analysis/heldout_eval.py --out runs/27b/heldout_long_sft --adapter $C/main $HELDOUT_ARGS || return 1
            python analysis/heldout_eval.py --compare runs/27b/heldout_base runs/27b/heldout_long_sft 2>&1 | grep -a -v -i "warn" > runs/27b/heldout_long_sft.compare.txt
        else
            echo "NOTE: runs/27b/heldout_base is not here - the held-out sets are left out."
        fi
    }
    stamp "final evaluation: GPU $GA the reference sets, GPU $GB the short benchmarks twice + held-out"
    lanes reference short; status=$?
    show 8
    [ $status -eq 0 ] || stop "final evaluation (see long_27b.gpu$GA.log, long_27b.gpu$GB.log)"
fi
finish "ALL DONE (stages: $STAGES)"
