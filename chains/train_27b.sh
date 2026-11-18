#!/usr/bin/env bash
# Step 3 for Qwen3.8-27B on two cards: training data, the two training stages, and the evaluation of the result.
#   data    stop points (code, tool, instruction), draft and give-up examples, shortest math solutions -> one training file
#   sft     LoRA on that file                                    -> checkpoints/27b/stoppoint_mixG
#   grpo    2-step smoke test, then 50 RL steps from that adapter -> checkpoints/27b/grpo
#   eval    the final adapter: two full benchmark runs, the held-out sets, comparisons with the base model
#   extras  the same evaluations for the supervised-only adapter
# Needs the output of step 2 in runs/27b (rollouts_*, base, heldout_base) and the environment .venv27.
#
#     nohup bash chains/train_27b.sh >> train_27b.log 2>&1 &        # or inside tmux
#     tail -f train_27b.log
#
# Two cards (GPUS="0 1", the default). The data and evaluation parts run as two lanes side by side, one per card,
# each with its own log (train_27b.gpu<N>.log); supervised training uses the first card; the RL stage uses both
# (vLLM generates on the first, the trained model sits on the second - 27B does not fit both on one card).
# Safe to stop and start again with the same command: finished parts are skipped, the stop search, the RL stage
# and the benchmark runs continue where they stopped. If a part fails the chain stops after its lane, says which
# log to look at, and still packs what exists.
# Optional environment:
#     BASE_MODEL=Qwen/Qwen3.8-27B           the model (default)
#     GPUS="0 1"                            the two cards
#     STAGES="data sft grpo eval extras"    which parts to run (default: all, in this order)
#     VENV=.venv27                          the Python environment
#   For a quick plumbing test only: GRPO_ARGS="--steps 2 --prompts 8 --group 4", EVAL_ARGS="--limit 3 --samples 2",
#   HELDOUT_ARGS="--limit 6", SFT_ARGS="--epochs 0.1".
# At the end it packs results_27b_step3.tar.gz (adapters, training data, evaluation records, logs) - send that back.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 1
source "${VENV:-.venv27}/bin/activate" || { echo "no environment: run chains/probe_27b.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
STAGES=${STAGES:-data sft grpo eval extras}
GRPO_ARGS=${GRPO_ARGS:---steps 50 --prompts 24 --group 8}
set -- ${GPUS:-0 1}
GA=$1 GB=${2:-$1}
R=runs/27b
C=checkpoints/27b
mkdir -p $R $C
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
        settle
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

finish() {
    python - <<'PY'
import json
from pathlib import Path
runs = Path("runs/27b")
for name in ("grpo_full1", "grpo_full2", "mixG_full"):
    path = runs / f"{name}.compare.txt"
    if path.exists():
        lines = path.read_text().splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("average")), None)
        if start is not None:
            print(f"{name} against the base model:")
            print(chr(10).join(lines[start:start + 3]))
log = Path("checkpoints/27b/grpo/log.jsonl")
if log.exists():
    steps = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    print(f"\nRL stage: {len(steps)} steps, {steps[-1]['minutes']:.0f} min" if steps else "")
PY
    tar czf results_27b_step3.tar.gz --exclude="$R/rollouts_*" --exclude="$R/base" --exclude="$R/heldout_base" \
        --exclude="$C/grpo/latest" --exclude="$C/grpo/optimizer.pt" --exclude="$C/grpo/step[1-4]0" --exclude="$C/grpo_smoke" \
        $R $C train_27b*.log 2>/dev/null
    stamp "$1 - send back results_27b_step3.tar.gz ($(du -h results_27b_step3.tar.gz | cut -f1))"
    exit "${2:-0}"
}
stop() { finish "STOPPED at: $1 (fix or report it, then start the same command again)" 1; }

# lanes <name A> <name B>: runs the functions lane_<A> on card A and lane_<B> on card B side by side, then
# returns 1 if either failed. With one card they run one after the other.
lanes() {
    if [ "$GA" != "$GB" ]; then
        ( export CUDA_VISIBLE_DEVICES=$GA; "lane_$1" ) >> train_27b.gpu$GA.log 2>&1 &
        local a=$!
        ( export CUDA_VISIBLE_DEVICES=$GB; "lane_$2" ) >> train_27b.gpu$GB.log 2>&1 &
        local b=$!
        wait $a; local sa=$?
        wait $b; local sb=$?
        return $(( sa || sb ))
    fi
    ( export CUDA_VISIBLE_DEVICES=$GA; "lane_$1" && "lane_$2" )
}
show() { grep -h -a "^\[" train_27b.gpu$GA.log train_27b.gpu$GB.log 2>/dev/null | sort | grep -a -v "start" | tail -${1:-12}; }

[ -z "$CHILD" ] && one_chain $R/.train.lock
stamp "step 3 start: model $BASE_MODEL, stages: $STAGES, cards: $GA $GB"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"
git status --short 2>/dev/null | grep -v '^??' | sed 's/^/uncommitted change: /'
for pool in math code tool constraint; do
    [ -s $R/rollouts_$pool/rollouts.jsonl ] || stop "runs/27b/rollouts_$pool is missing (step 2 output)"
    [ -s data/pool27b_$pool.jsonl ] || stop "data/pool27b_$pool.jsonl is missing (qwen27b_pools.tar.gz)"
done
[ -s data/wrapper_train.jsonl ] || stop "data/wrapper_train.jsonl is missing (it came with the 4B data archive)"
for gpu in $GA $GB; do
    nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader > /dev/null 2>&1 || stop "GPU $gpu does not exist here (nvidia-smi -L)"
done
# vLLM alone on the first card, the trained model on the second
TWO="--train-device cuda:1 --vllm-memory 0.85"
if has grpo && [ "$GA" = "$GB" ]; then
    [ -n "$ONE_CARD_TEST" ] || stop "the RL stage of a 27B model needs two cards (GPUS=\"0 1\")"
    TWO="--vllm-memory 0.35"        # plumbing tests with a small model only
fi
# Model-written code must run here (stop search, draft, give-up, the RL reward): check the sandbox for real.
sandbox=$(python -c "import sys; sys.path.insert(0, '.'); from stopping.code_tasks import self_test; print(self_test() or 'OK')" 2>&1 | tail -1)
echo "code sandbox self-test: $sandbox"
[ "$sandbox" = "OK" ] || stop "the code sandbox does not work here (see the self-test line above)"

if has data; then
    # Step 2 graded code with the broken sandbox (every program failed): grade the code answers again.
    step "regrade code rollouts" $R/rollouts_code/regraded.txt $R/regrade_code.log \
        bash -c "python stopping/regrade_rollouts.py $R/rollouts_code && date > $R/rollouts_code/regraded.txt" || stop "regrade code rollouts"
    step "regrade held-out base" $R/heldout_base/regraded.txt $R/regrade_heldout.log \
        bash -c "python analysis/heldout_eval.py --regrade $R/heldout_base && date > $R/heldout_base/regraded.txt" || stop "regrade held-out base"
    grep -a "strict\|knows" $R/regrade_code.log | tail -3

    lane_data_code() {
        step "stop search code" $R/stop_points_code/train.jsonl $R/stop_points_code.log \
            python stopping/find_stop_points.py $R/rollouts_code --out $R/stop_points_code \
            --answer-tokens 1024 --max-rollout-tokens 6144 || return 1
        step "draft code stops" $R/verified_code_draft/train.jsonl $R/verified_code_draft.log \
            python stopping/verified_examples.py $R/rollouts_code $R/stop_points_code --mode draft \
            --out $R/verified_code_draft --answer-tokens 1024 --max-rollout-tokens 6144 || return 1
        step "give-up code" $R/give_up_code/train.jsonl $R/give_up_code.log \
            python stopping/give_up_examples.py $R/rollouts_code --out $R/give_up_code --budget 2048 --answer-tokens 1024 \
            --max-rollout-tokens 6144 --max-examples 600
    }
    lane_data_rest() {
        step "stop search tool" $R/stop_points_tool/train.jsonl $R/stop_points_tool.log \
            python stopping/find_stop_points.py $R/rollouts_tool --out $R/stop_points_tool --answer-tokens 256 || return 1
        step "stop search instruction" $R/stop_points_constraint/train.jsonl $R/stop_points_constraint.log \
            python stopping/find_stop_points.py $R/rollouts_constraint --out $R/stop_points_constraint \
            --answer-tokens 1024 --max-rollout-tokens 4096 || return 1
        step "give-up math" $R/give_up_math/train.jsonl $R/give_up_math.log \
            python stopping/give_up_examples.py $R/rollouts_math --out $R/give_up_math --budget 2048 --answer-tokens 384 \
            --max-rollout-tokens 4096 --max-examples 300 || return 1
        step "shortest math solutions" $R/sft_shortest.jsonl $R/sft_shortest.log \
            python stopping/build_shortest.py $R/rollouts_math --out $R/sft_shortest.jsonl
    }
    stamp "training data: code on GPU $GA, tool / instruction / math on GPU $GB (details in train_27b.gpu<N>.log)"
    lanes data_code data_rest; status=$?
    show 14
    [ $status -eq 0 ] || stop "training data (see train_27b.gpu$GA.log and train_27b.gpu$GB.log)"
    step "training file" $R/train_mixG.jsonl $R/train_mixG.build.log \
        python stopping/build_mix.py --runs $R --out $R/train_mixG.jsonl || stop "training file"
    tail -1 $R/train_mixG.build.log
fi

if has sft; then
    export CUDA_VISIBLE_DEVICES=$GA
    step "LoRA SFT" $C/stoppoint_mixG/adapter_model.safetensors $R/train_mixG.log \
        python stopping/train_lora.py --data $R/train_mixG.jsonl --out $C/stoppoint_mixG --max-length 7400 $SFT_ARGS || stop "LoRA SFT"
    grep -a -E "examples \(|^step " $R/train_mixG.log | sed -n '1p;$p'
    unset CUDA_VISIBLE_DEVICES
fi

POOLS="--pool math=data/pool27b_math.jsonl --pool code=data/pool27b_code.jsonl --pool tool=data/pool27b_tool.jsonl
       --pool instruction=data/pool27b_constraint.jsonl+data/wrapper_train.jsonl"
if has grpo; then
    if [ "$GA" = "$GB" ]; then export CUDA_VISIBLE_DEVICES=$GA; else export CUDA_VISIBLE_DEVICES=$GA,$GB; fi
    step "RL smoke test (2 steps)" $C/grpo_smoke/adapter_model.safetensors $R/grpo_smoke.log \
        python stopping/train_grpo.py --init $C/stoppoint_mixG --out $C/grpo_smoke --steps 2 --prompts 8 --group 4 $TWO $POOLS \
        || stop "RL smoke test"
    resume=""
    [ -s $C/grpo/log.jsonl ] && [ -d $C/grpo/latest ] && resume="--resume" && stamp "RL stage: continuing after step $(wc -l < $C/grpo/log.jsonl)"
    step "RL stage" $C/grpo/adapter_model.safetensors $R/grpo.log \
        python stopping/train_grpo.py --init $C/stoppoint_mixG --out $C/grpo $GRPO_ARGS $TWO $POOLS $resume || stop "RL stage"
    grep -a -E '^\{"step"' $R/grpo.log | sed -n '1p;10p;20p;30p;40p;$p' | cut -c1-230
    unset CUDA_VISIBLE_DEVICES
fi

full() {     # full <adapter directory> <run name>
    step "benchmarks $2" $R/$2/results.json $R/$2.log \
        python -m evaluation.run --out $R/$2 --model "$BASE_MODEL" --adapter $1 $EVAL_ARGS || return 1
    python -m evaluation.compare $R/base $R/$2 --json $R/$2.compare.json > $R/$2.compare.txt 2>&1
    grep -A2 "^average" $R/$2.compare.txt
    python analysis/hard_tail.py $R/base $R/$2 > $R/$2.tail.txt 2>&1
}
heldout() {  # heldout <adapter directory> <name>
    step "held-out $2" $R/heldout_$2/records.jsonl $R/heldout_$2.log \
        python analysis/heldout_eval.py --out $R/heldout_$2 --adapter $1 $HELDOUT_ARGS || return 1
    { python analysis/heldout_eval.py --compare $R/heldout_base $R/heldout_$2
      python analysis/hard_tail.py $R/heldout_base $R/heldout_$2
      python analysis/difficulty_bins.py $R/heldout_base $R/heldout_$2 --levels --metric knows --json $R/heldout_$2.ladder.json
    } 2>&1 | grep -a -v -i "warn" > $R/heldout_$2.compare.txt
    grep -a -A2 "^humaneval\|^gsm8k_test\|^math500" $R/heldout_$2.compare.txt | head -12
}

if has eval || has extras; then
    [ -s $R/base/results.json ] || stop "runs/27b/base is missing (step 2 output)"
    [ -s $R/heldout_base/regraded.txt ] || stop "runs/27b/heldout_base is not graded again yet (run the data stage first)"
    lane_eval_a() {
        if has eval; then full $C/grpo grpo_full1 || return 1; heldout $C/grpo grpo || return 1; fi
        if has extras; then heldout $C/stoppoint_mixG mixG || return 1; fi
    }
    lane_eval_b() {
        if has eval; then full $C/grpo grpo_full2 || return 1; fi
        if has extras; then full $C/stoppoint_mixG mixG_full || return 1; fi
    }
    stamp "evaluation: GPU $GA runs grpo_full1 + held-out, GPU $GB runs grpo_full2 + the supervised-only adapter"
    lanes eval_a eval_b; status=$?
    show 10
    for name in grpo_full1 grpo_full2 mixG_full; do
        [ -s $R/$name.compare.txt ] && { echo "$name:"; grep -A2 "^average" $R/$name.compare.txt; }
    done
    [ $status -eq 0 ] || stop "evaluation (see train_27b.gpu$GA.log and train_27b.gpu$GB.log)"
    stamp "MAIN RESULT DONE"
fi
finish "ALL DONE (stages: $STAGES)"
