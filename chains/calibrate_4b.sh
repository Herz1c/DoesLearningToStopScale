#!/usr/bin/env bash
# Difficulty calibration for Qwen3-4B: how often does the base model answer right on each difficulty
# tier of each training domain? 12 runs (4 domains x 3 tiers), 200 questions x 4 samples each, then
# the mixing rule. No training, no evaluation benchmark.
#
#     bash chains/setup.sh                                              # once
#     nohup bash chains/calibrate_4b.sh > calibrate_4b.log 2>&1 &       # or inside tmux
#     tail -f calibrate_4b.log
#
# Safe to stop and start again: finished questions are kept, each run continues where it stopped.
# Optional environment:
#     BASE_MODEL=Qwen/Qwen3-4B     the model (default)
#     ONLY="math tool"             only these domains (default: math code tool constraint) - use it to
#                                  split the work over several GPUs, e.g. on a second card:
#                                  CUDA_VISIBLE_DEVICES=1 ONLY=code bash chains/calibrate_4b.sh
#     QUESTIONS=200                questions per tier
# When it ends it writes calibration_4b_results.tar.gz (summaries, logs and the rollouts) - send that back.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate || exit 1
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B} PYTHONUNBUFFERED=1
ONLY=${ONLY:-math code tool constraint}
N=${QUESTIONS:-200}
OUT=runs/cal4b
mkdir -p $OUT
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# domain  tier  data file  source  generation cap
RUNS="
math       t0 data/train.jsonl              gsm8k             4096
math       t1 data/math_hard_train.jsonl    math              4096
math       t2 data/math_l5_train.jsonl      math              4096
code       t0 data/code_train.jsonl         kodcode           6144
code       t1 data/code_hard_train.jsonl    kodcode           6144
code       t2 data/code_hardest_train.jsonl kodcode           6144
tool       t0 data/tool_train_clean.jsonl   glaive            4096
tool       t1 data/tool_wide.jsonl          glaive_wide       4096
tool       t2 data/tool_removed.jsonl       glaive_removed    4096
constraint t0 data/constraint_train.jsonl   dolly_constraints 4096
constraint t1 data/constraint_c3.jsonl      dolly_constraints 4096
constraint t2 data/constraint_c4.jsonl      dolly_constraints 4096
"
# Targets: the share of rollouts in which the Qwen3-0.6B base knew the answer on its original pools.
target() { case $1 in math) echo 0.708;; code) echo 0.519;; tool) echo 0.632;; constraint) echo 0.626;; esac; }

stamp "calibration start: model $BASE_MODEL, domains: $ONLY, $N questions x 4 per tier"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "$RUNS" | while read -r domain tier data source cap; do
    [ -z "$domain" ] && continue
    case " $ONLY " in *" $domain "*) ;; *) continue;; esac
    name=${domain}_$tier
    stamp "$name start ($data, cap $cap)"
    start=$(date +%s)
    python stopping/generate_rollouts.py --out $OUT/$name --data $data --sources $source --questions $N \
        --max-tokens $cap >> $OUT/$name.log 2>&1 < /dev/null
    status=$?
    grep -a "questions, \|questions done" $OUT/$name.log | tail -1
    if [ $status -ne 0 ] || [ ! -s $OUT/$name/rollouts.jsonl ]; then
        stamp "$name FAILED (exit $status) - last lines of $OUT/$name.log:"; tail -5 $OUT/$name.log
    else
        stamp "$name done in $(( ($(date +%s) - start) / 60 )) min"
    fi
done

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
out = Path("runs/cal4b")
print("\ndomain      tier  knows   strict  capped  | chosen mix")
for path in sorted(out.glob("*.json")):
    r = json.loads(path.read_text())
    for k, (tier, p) in enumerate(r["tiers"].items()):
        mix = (f"{1 - r['hard_fraction']:.0%} {Path(r['easy_tier']).name} + {r['hard_fraction']:.0%} {Path(r['hard_tier']).name}"
               f" -> {r['predicted_knows']:.1%} (target {r['target']:.1%}{'' if r['target_reached'] else ', NOT reached'})") if k == 0 else ""
        print(f"{path.stem:11s} {Path(tier).name[-2:]}   {p['knows']:6.1%}  {p['strict']:6.1%}  {p['capped']:6.1%}  | {mix}")
PY
tar czf calibration_4b_results.tar.gz $OUT calibrate_4b.log 2>/dev/null || tar czf calibration_4b_results.tar.gz $OUT
stamp "ALL DONE - send back calibration_4b_results.tar.gz ($(du -h calibration_4b_results.tar.gz | cut -f1))"
