#!/usr/bin/env bash
# Repeats the forced answers of the CODE questions of the long-reasoning calibration (chains/calibrate_long_27b.sh).
# In the first run the code judge marked right programs as wrong while the engine was loaded (time limits), so
# the code part of the readouts cannot be used. This time the forced answers are kept in the files and judged
# after the engine has gone; they can also be judged again anywhere else.
#
#     nohup bash chains/recalibrate_code_27b.sh >> recalibrate_code_27b.log 2>&1 &        # or inside tmux
#     tail -f recalibrate_code_27b.log
#
# Needs: the environment .venv27 and runs/27b_long_cal/rollouts.*.jsonl of the first run (nothing is generated
# again there; the rollouts are only judged again). No internet, no new data.
# Steps: 1. judge self-test   2. the rollouts' own programs judged again   3. forced answers of the code rollouts
# (every card in GPUS takes a share, default "0 1")   4. the kept answers judged, engine not loaded.
# Safe to stop and start again: finished rollouts are kept. Optional: GPUS="0 1"; JUDGE_WORKERS=<n> (parallel
# judge processes; default = the cores of this job, at most 12).
# When it ends it packs results_27b_long_cal_code.tar.gz - send that back.
cd "$(dirname "$0")/.." || exit 1
source "${VENV:-.venv27}/bin/activate" || { echo "no environment: run chains/probe_27b.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
CARDS=(${GPUS:-0 1})
N=${#CARDS[@]}
R=runs/27b_long_cal
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
source chains/gpu_wait.sh
one_chain $R/.recalibrate.lock

stamp "code readouts again: model $BASE_MODEL, cards: ${CARDS[*]}"
ls $R/rollouts*.jsonl > /dev/null 2>&1 || { stamp "$R/rollouts*.jsonl are missing: this chain needs the first run's folder"; exit 1; }
echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"
echo "cores: node $(nproc --all), this job $(nproc), judge workers $(python -c "import sys; sys.path.insert(0, '.'); from stopping.long_tasks import judge_workers; print(judge_workers())" 2>/dev/null)"
sandbox=$(python -c "import sys; sys.path.insert(0, '.'); from stopping.code_tasks import self_test; print(self_test() or 'OK')" 2>&1 | tail -1)
echo "code sandbox self-test: $sandbox"
python stopping/long_tasks.py --judge-timing 2>&1 | grep -a "judge timing"

# the first run's readouts are kept aside once; the working files keep every row except the code ones
if [ ! -d $R/readouts_v1 ]; then
    mkdir -p $R/readouts_v1 && cp -p $R/readouts*.jsonl $R/readouts_v1/ 2>/dev/null
    python - <<'PY'
import glob, json
for path in sorted(glob.glob("runs/27b_long_cal/readouts*.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip().endswith("}")]
    kept = [r for r in rows if r["task"] != "code_io"]
    open(path, "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{path}: {len(rows)} rows, {len(rows) - len(kept)} code rows set aside (copy in readouts_v1)")
PY
fi

stamp "rollouts: their own programs judged again (engine not loaded)"
python stopping/readouts.py $R --regrade 2>&1 | grep -a -v -i "warn" | tee $R/regrade_rollouts.log

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
for k in "${!CARDS[@]}"; do
    ( export CUDA_VISIBLE_DEVICES=${CARDS[$k]}
      job code_readouts$k python stopping/readouts.py $R --shard $k/$N --tasks code_io --judge-later $READOUT_ARGS
    ) > recalibrate_code_27b.gpu${CARDS[$k]}.log 2>&1 &
done
wait
for k in "${!CARDS[@]}"; do cat recalibrate_code_27b.gpu${CARDS[$k]}.log; grep -a "with the engine loaded" -A2 $R/code_readouts$k.log | grep -a "judge timing"; done

stamp "kept answers judged (engine not loaded)"
python stopping/readouts.py $R --regrade 2>&1 | grep -a -v -i "warn" | tee $R/regrade_readouts.log
stamp "judging done"
echo; echo "== from where the answer is settled"
python stopping/readouts.py $R --report 2>/dev/null | tee $R/readouts_report_v2.txt
tar czf results_27b_long_cal_code.tar.gz $R/readouts*.jsonl $R/readouts_v1 $R/*.log $R/*.txt recalibrate_code_27b*.log 2>/dev/null
stamp "ALL DONE - send back results_27b_long_cal_code.tar.gz ($(du -h results_27b_long_cal_code.tar.gz | cut -f1))"
