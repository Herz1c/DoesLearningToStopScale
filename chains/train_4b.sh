#!/usr/bin/env bash
# Step 3 for Qwen3-4B: training data, the two training stages, and the evaluation of the result.
#   data    stop points (code, tool, instruction), draft and give-up examples, shortest math solutions -> one training file
#   sft     LoRA on that file                                    -> checkpoints/4b/stoppoint_mixG
#   grpo    2-step smoke test, then 50 RL steps from that adapter -> checkpoints/4b/grpo
#   eval    the final adapter: two full benchmark runs, the held-out sets, comparisons with the base model
#   extras  the same evaluations for the supervised-only adapter
# Needs the output of step 2 in runs/4b (rollouts_*, base, heldout_base).
#
#     SKIP_INSTALL=1 bash chains/setup.sh
#     nohup bash chains/train_4b.sh >> train_4b.log 2>&1 &         # or inside tmux
#     tail -f train_4b.log
#
# Safe to stop and start again with the same command: finished parts are skipped, the stop search, the RL stage
# and the benchmark runs continue where they stopped. If a part fails the chain stops there, says which log to
# look at, and still packs what exists.
# Optional environment:
#     BASE_MODEL=Qwen/Qwen3-4B              the model (default)
#     STAGES="data sft grpo eval extras"    which parts to run (default: all, in this order)
#   For a quick plumbing test only: GRPO_ARGS="--steps 2 --prompts 8 --group 4", EVAL_ARGS="--limit 3 --samples 2",
#   HELDOUT_ARGS="--limit 6".
# At the end it packs results_4b_step3.tar.gz (adapters, training data, evaluation records, logs) - send that back.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate || exit 1
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B} PYTHONUNBUFFERED=1
STAGES=${STAGES:-data sft grpo eval extras}
GRPO_ARGS=${GRPO_ARGS:---steps 50 --prompts 24 --group 8}
R=runs/4b
C=checkpoints/4b
mkdir -p $R $C
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }

# step <label> <file that proves the part is done> <log> <command...>
step() {
    local label=$1 proof=$2 log=$3; shift 3
    if [ -s "$proof" ]; then stamp "$label: already done, skipped"; return 0; fi
    stamp "$label start"
    local start; start=$(date +%s)
    "$@" >> "$log" 2>&1 < /dev/null
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
runs = Path("runs/4b")
for name in ("grpo_full1", "grpo_full2", "mixG_full"):
    path = runs / f"{name}.compare.txt"
    if path.exists():
        lines = path.read_text().splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("average")), None)
        if start is not None:
            print(f"{name} against the base model:")
            print(chr(10).join(lines[start:start + 3]))
log = Path("checkpoints/4b/grpo/log.jsonl")
if log.exists():
    steps = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    print(f"\nRL stage: {len(steps)} steps, {steps[-1]['minutes']:.0f} min" if steps else "")
PY
    tar czf results_4b_step3.tar.gz --exclude="$R/rollouts_*" --exclude="$R/base" --exclude="$R/heldout_base" \
        --exclude="$C/grpo/latest" --exclude="$C/grpo/optimizer.pt" --exclude="$C/grpo_smoke" \
        $R $C train_4b.log 2>/dev/null
    stamp "$1 - send back results_4b_step3.tar.gz ($(du -h results_4b_step3.tar.gz | cut -f1))"
    exit "${2:-0}"
}
stop() { finish "STOPPED at: $1 (fix or report it, then start the same command again)" 1; }

stamp "step 3 start: model $BASE_MODEL, stages: $STAGES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse --short HEAD 2>/dev/null | sed 's/^/code at commit /'
git status --short 2>/dev/null | grep -v '^??' | sed 's/^/uncommitted change: /'
for pool in math code tool constraint; do
    [ -s $R/rollouts_$pool/rollouts.jsonl ] || stop "runs/4b/rollouts_$pool is missing (step 2 output)"
done

if has data; then
    # math: the strict grade is now the bare closing line "Final answer: N"; the stored grades are older
    step "regrade math rollouts" $R/rollouts_math/regraded.txt $R/regrade_math.log \
        bash -c "python stopping/regrade_rollouts.py $R/rollouts_math && date > $R/rollouts_math/regraded.txt" || stop "regrade"
    step "stop search code" $R/stop_points_code/train.jsonl $R/stop_points_code.log \
        python stopping/find_stop_points.py $R/rollouts_code --out $R/stop_points_code \
        --answer-tokens 1024 --max-rollout-tokens 6144 || stop "stop search code"
    step "stop search tool" $R/stop_points_tool/train.jsonl $R/stop_points_tool.log \
        python stopping/find_stop_points.py $R/rollouts_tool --out $R/stop_points_tool --answer-tokens 256 || stop "stop search tool"
    step "stop search instruction" $R/stop_points_constraint/train.jsonl $R/stop_points_constraint.log \
        python stopping/find_stop_points.py $R/rollouts_constraint --out $R/stop_points_constraint \
        --answer-tokens 1024 --max-rollout-tokens 4096 || stop "stop search instruction"
    step "draft code stops" $R/verified_code_draft/train.jsonl $R/verified_code_draft.log \
        python stopping/verified_examples.py $R/rollouts_code $R/stop_points_code --mode draft \
        --out $R/verified_code_draft --answer-tokens 1024 --max-rollout-tokens 6144 || stop "draft code stops"
    step "give-up code" $R/give_up_code/train.jsonl $R/give_up_code.log \
        python stopping/give_up_examples.py $R/rollouts_code --out $R/give_up_code --budget 2048 --answer-tokens 1024 \
        --max-rollout-tokens 6144 --max-examples 600 || stop "give-up code"
    step "give-up math" $R/give_up_math/train.jsonl $R/give_up_math.log \
        python stopping/give_up_examples.py $R/rollouts_math --out $R/give_up_math --budget 2048 --answer-tokens 384 \
        --max-rollout-tokens 4096 --max-examples 300 || stop "give-up math"
    step "shortest math solutions" $R/sft_shortest.jsonl $R/sft_shortest.log \
        python stopping/build_shortest.py $R/rollouts_math --out $R/sft_shortest.jsonl || stop "shortest math solutions"
    step "training file" $R/train_mixG.jsonl $R/train_mixG.build.log \
        python stopping/build_mix.py --runs $R --out $R/train_mixG.jsonl || stop "training file"
    tail -1 $R/train_mixG.build.log
fi

if has sft; then
    step "LoRA SFT" $C/stoppoint_mixG/adapter_model.safetensors $R/train_mixG.log \
        python stopping/train_lora.py --data $R/train_mixG.jsonl --out $C/stoppoint_mixG --max-length 7400 || stop "LoRA SFT"
    grep -a -E "examples \(|^step " $R/train_mixG.log | sed -n '1p;$p'
fi

POOLS="--pool math=data/pool4b_math.jsonl --pool code=data/pool4b_code.jsonl --pool tool=data/pool4b_tool.jsonl
       --pool instruction=data/pool4b_constraint.jsonl+data/wrapper_train.jsonl"
if has grpo; then
    step "RL smoke test (2 steps)" $C/grpo_smoke/adapter_model.safetensors $R/grpo_smoke.log \
        python stopping/train_grpo.py --init $C/stoppoint_mixG --out $C/grpo_smoke --steps 2 --prompts 8 --group 4 $POOLS \
        || stop "RL smoke test"
    resume=""
    [ -s $C/grpo/log.jsonl ] && [ -d $C/grpo/latest ] && resume="--resume" && stamp "RL stage: continuing after step $(wc -l < $C/grpo/log.jsonl)"
    step "RL stage" $C/grpo/adapter_model.safetensors $R/grpo.log \
        python stopping/train_grpo.py --init $C/stoppoint_mixG --out $C/grpo $GRPO_ARGS $POOLS $resume || stop "RL stage"
    grep -a -E '^\{"step"' $R/grpo.log | sed -n '1p;10p;20p;30p;40p;$p' | cut -c1-230
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

if has eval; then
    [ -s $R/base/results.json ] || stop "runs/4b/base is missing (step 2 output)"
    full $C/grpo grpo_full1 || stop "benchmarks grpo_full1"
    full $C/grpo grpo_full2 || stop "benchmarks grpo_full2"
    heldout $C/grpo grpo || stop "held-out grpo"
    stamp "MAIN RESULT DONE"
fi
if has extras; then
    full $C/stoppoint_mixG mixG_full || stop "benchmarks mixG_full"
    heldout $C/stoppoint_mixG mixG || stop "held-out mixG"
fi
finish "ALL DONE (stages: $STAGES)"
