#!/usr/bin/env bash
# Difficulty calibration for Qwen3.8-27B: how often does the base model answer right on each difficulty
# tier of each training domain? 12 runs (4 domains x 3 tiers), 200 questions x 4 samples each, then
# the mixing rule. No training, no evaluation benchmark.
#
#     nohup bash chains/calibrate_27b.sh >> calibrate_27b.log 2>&1 &     # or inside tmux
#     tail -f calibrate_27b.log
#
# Needs: the environment .venv27 that chains/probe_27b.sh built, the tier files of qwen27b_calibration.tar.gz
# unpacked in the repository (data/...), and bubblewrap (bwrap) for the code domain - without it the code
# domain is skipped and the other three still run.
# Safe to stop and start again: finished questions are kept, each run continues where it stopped.
# Optional environment:
#     BASE_MODEL=Qwen/Qwen3.8-27B  the model (default)
#     ONLY="math tool"             only these domains (default: math code tool constraint) - use it to
#                                  split the work over several GPUs, e.g. on a second card:
#                                  CUDA_VISIBLE_DEVICES=1 ONLY=code bash chains/calibrate_27b.sh
#     SPLIT_GPUS="0 1"             two cards at once: code on the first, the other three domains on the second
#     QUESTIONS=200                questions per tier
#     VENV=.venv27                 the Python environment
# When it ends it writes calibration_27b_results.tar.gz (summaries, logs and the rollouts) - send that back.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 1
source "${VENV:-.venv27}/bin/activate" || { echo "no environment: run chains/probe_27b.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
ONLY=${ONLY:-math code tool constraint}
N=${QUESTIONS:-200}
OUT=runs/cal27b
mkdir -p $OUT
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
source chains/gpu_wait.sh

# domain  tier  data file  source  generation cap
# math: GSM8K, MATH levels 3-4, MATH level 5.  code: KodCode easy, then the hard pool split by KodCode's own
# pass rate (above / at most 0.12).  tool: plain, more functions offered, the right function removed.
# constraint: one or two, three, four format constraints.  Tier 0 of every domain is the tier 0 of the 4B run.
RUNS="
math       t0 data/train.jsonl              gsm8k             4096
math       t1 data/math_l34_train.jsonl     math              4096
math       t2 data/math_l5_train.jsonl      math              4096
code       t0 data/code_train.jsonl         kodcode           6144
code       t1 data/code27_t1.jsonl          kodcode           6144
code       t2 data/code27_t2.jsonl          kodcode           6144
tool       t0 data/tool_train_fixed.jsonl   glaive            4096
tool       t1 data/tool_wide_fixed.jsonl    glaive_wide       4096
tool       t2 data/tool_removed_fixed.jsonl glaive_removed    4096
constraint t0 data/constraint_train.jsonl   dolly_constraints 4096
constraint t1 data/constraint_c3.jsonl      dolly_constraints 4096
constraint t2 data/constraint_c4.jsonl      dolly_constraints 4096
"
# Targets: the share of rollouts in which the Qwen3-0.6B base knew the answer on its original pools
# (tool and constraint with the label and checker fixes of the 4B study).
target() { case $1 in math) echo 0.708;; code) echo 0.519;; tool) echo 0.637;; constraint) echo 0.666;; esac; }

stamp "calibration start: model $BASE_MODEL, domains: $ONLY, $N questions x 4 per tier"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"; git status --short 2>/dev/null | head -5
echo "$RUNS" | while read -r domain tier data source cap; do [ -n "$data" ] && [ ! -s "$data" ] && echo "WARNING: $data is missing"; done
if ! bwrap --unshare-all --ro-bind / / true 2>/dev/null; then
    echo "WARNING: bubblewrap (bwrap) cannot start a sandbox here: the code domain is skipped (its answers cannot be run safely)"
    ONLY=$(echo " $ONLY " | sed 's/ code / /')
fi
# Two cards: SPLIT_GPUS="0 1" runs the code domain on the first card and the other three on the second, side by
# side (they are about equal in work), each with its own log, then the mixing rule and the summary once.
if [ -n "$SPLIT_GPUS" ]; then
    set -- $SPLIT_GPUS
    stamp "two cards: code on GPU $1 (calibrate_27b.gpu$1.log), math tool constraint on GPU $2 (calibrate_27b.gpu$2.log)"
    CUDA_VISIBLE_DEVICES=$1 ONLY="code" SPLIT_GPUS= RUNS_ONLY=1 bash "$SELF" > calibrate_27b.gpu$1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$2 ONLY="math tool constraint" SPLIT_GPUS= RUNS_ONLY=1 bash "$SELF" > calibrate_27b.gpu$2.log 2>&1 &
    wait
    grep -h -a "done in\|FAILED\|WARNING" calibrate_27b.gpu$1.log calibrate_27b.gpu$2.log
else
echo "$RUNS" | while read -r domain tier data source cap; do
    [ -z "$domain" ] && continue
    case " $ONLY " in *" $domain "*) ;; *) continue;; esac
    name=${domain}_$tier
    stamp "$name start ($data, cap $cap)"
    start=$(date +%s)
    for attempt in 1 2; do      # a second try continues with the questions that are not done yet
        settle
        python stopping/generate_rollouts.py --out $OUT/$name --data $data --sources $source --questions $N \
            --max-tokens $cap >> $OUT/$name.log 2>&1 < /dev/null
        status=$?
        [ $status -eq 0 ] && break
        [ $attempt -eq 1 ] && stamp "$name: first try failed (exit $status), trying once more"
    done
    grep -a "questions, \|questions done" $OUT/$name.log | tail -1
    if [ $status -ne 0 ] || [ ! -s $OUT/$name/rollouts.jsonl ]; then
        stamp "$name FAILED (exit $status) - last lines of $OUT/$name.log:"; tail -5 $OUT/$name.log
    else
        stamp "$name done in $(( ($(date +%s) - start) / 60 )) min"
    fi
done
fi
[ -n "$RUNS_ONLY" ] && { stamp "runs of this card finished"; exit 0; }

for domain in $ONLY; do
    tiers=""
    for tier in t0 t1 t2; do [ -s $OUT/${domain}_$tier/rollouts.jsonl ] && tiers="$tiers $OUT/${domain}_$tier"; done
    if [ "$(echo $tiers | wc -w)" -eq 3 ]; then
        stamp "$domain: mixing rule (target $(target $domain))"
        python stopping/mix_pools.py ladder --tiers $tiers --target "$(target $domain)" --out $OUT/$domain.json \
            > $OUT/$domain.ladder.log 2>&1 || { stamp "$domain ladder FAILED"; tail -3 $OUT/$domain.ladder.log; }
    else
        stamp "$domain: not all three tiers finished, no mixing rule"
    fi
done

python - <<'PY'
import json
from pathlib import Path
out = Path("runs/cal27b")
print("\ndomain      tier  knows   strict  capped  | chosen mix")
for path in sorted(out.glob("*.json")):
    r = json.loads(path.read_text())
    for k, (tier, p) in enumerate(r["tiers"].items()):
        mix = (f"{1 - r['hard_fraction']:.0%} {Path(r['easy_tier']).name} + {r['hard_fraction']:.0%} {Path(r['hard_tier']).name}"
               f" -> {r['predicted_knows']:.1%} (target {r['target']:.1%}{'' if r['target_reached'] else ', NOT reached'})") if k == 0 else ""
        print(f"{path.stem:11s} {Path(tier).name[-2:]}   {p['knows']:6.1%}  {p['strict']:6.1%}  {p['capped']:6.1%}  | {mix}")
PY
tar czf calibration_27b_results.tar.gz $OUT calibrate_27b*.log 2>/dev/null || tar czf calibration_27b_results.tar.gz $OUT
stamp "ALL DONE - send back calibration_27b_results.tar.gz ($(du -h calibration_27b_results.tar.gz | cut -f1))"
