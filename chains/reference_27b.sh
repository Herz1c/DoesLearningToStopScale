#!/usr/bin/env bash
# Long-reasoning benchmarks for Qwen3.8-27B: the base model and the trained adapter on the public sets of
# evaluation/reference.py (AIME 2026, HMMT, GPQA-Diamond, LiveCodeBench v6, IFBench, 1 000 MMLU-Pro questions),
# at temperature 1.0 with room for up to 253 952 generated tokens. Generation only, no training.
#
#     nohup bash chains/reference_27b.sh >> reference_27b.log 2>&1 &        # or inside tmux
#     tail -f reference_27b.log
#
# Two cards (GPUS="0 1", the default): the base model on the first, the adapter on the second, side by side,
# each with its own log (reference_27b.gpu<N>.log). With one card (GPUS="0") they run one after the other.
# Needs: the environment .venv27, the adapter checkpoints/27b/grpo (from the training run), internet.
# GPQA-Diamond is behind a login: accept its terms at https://huggingface.co/datasets/Idavidrein/gpqa and log in
# (`hf auth login`, or HF_TOKEN in the environment). Without that the set is left out and everything else runs.
# Safe to stop and start again: finished sets are kept, an unfinished set starts over.
# Optional environment:
#     GPUS="0 1"                         the cards
#     RUNS="base grpo"                   which models (add "theirs" for bottlecapai/ThinkingCap-Qwen3.8-27B, a
#                                        gated model: accept its terms on the Hub first; it runs after the others)
#     SAMPLES=4                          replies per question (MMLU-Pro: always 1)
#     SETS=aime26,hmmt_feb26,...         a subset of the sets
#   For a quick plumbing test only: REF_ARGS="--limit 3 --max-tokens 3000".
# When it ends it packs results_27b_reference.tar.gz (all replies, grades, logs) - send that back.
cd "$(dirname "$0")/.." || exit 1
source "${VENV:-.venv27}/bin/activate" || { echo "no environment: run chains/probe_27b.sh first"; exit 1; }
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
RUNS=${RUNS:-base grpo}
set -- ${GPUS:-0 1}
GA=$1 GB=${2:-$1}
R=runs/27b_reference
ADAPTER=${ADAPTER:-checkpoints/27b/grpo}
THEIRS=${THEIRS:-bottlecapai/ThinkingCap-Qwen3.8-27B}
mkdir -p $R
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
has() { case " $RUNS " in *" $1 "*) return 0;; esac; return 1; }
source chains/gpu_wait.sh
one_chain $R/.reference.lock

stamp "reference benchmarks start: model $BASE_MODEL, runs: $RUNS, cards: $GA $GB, samples ${SAMPLES:-4}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"
if ! python -c "import math_verify, emoji, syllapy, pandas, pyarrow, langdetect, nltk, immutabledict, absl" 2>/dev/null; then
    stamp "installing the graders' packages"
    UV=$(command -v uv || echo "$PWD/.venv_uv/bin/uv")
    "$UV" pip install --python "$VIRTUAL_ENV/bin/python" "math-verify[antlr4_13_2]" emoji syllapy pandas pyarrow \
        absl-py langdetect nltk immutabledict > $R/install.log 2>&1 || { stamp "install FAILED - last lines of $R/install.log:"; tail -5 $R/install.log; }
fi
stamp "questions: download and fix (pinned sources)"
python -m evaluation.reference --prepare 2>&1 | grep -a -v -i "warn" | tee $R/prepare.log
[ -s bench_reference/questions.jsonl ] || { stamp "no questions - see $R/prepare.log"; exit 1; }
grep -q "NOT AVAILABLE: gpqa" $R/prepare.log && echo "NOTE: GPQA-Diamond is left out (no access). To add it later: accept its terms, log in, start this chain again."
has grpo && [ ! -s $ADAPTER/adapter_model.safetensors ] && { stamp "the adapter $ADAPTER is missing (it is the result of chains/train_27b.sh)"; exit 1; }
sandbox=$(python -c "import sys; sys.path.insert(0, '.'); from stopping.code_tasks import self_test; print(self_test() or 'OK')" 2>&1 | tail -1)
echo "code sandbox self-test: $sandbox   (if not OK, the LiveCodeBench programs are graded later, elsewhere)"

# run <name> <extra arguments...>: one model on the card in CUDA_VISIBLE_DEVICES; two tries
run() {
    local name=$1 attempt; shift
    stamp "$name start (GPU $CUDA_VISIBLE_DEVICES)"
    local start; start=$(date +%s)
    for attempt in 1 2; do
        settle
        python -m evaluation.reference --out $R/$name --samples ${SAMPLES:-4} ${SETS:+--sets $SETS} $REF_ARGS "$@" \
            >> $R/$name.log 2>&1 < /dev/null && break
        [ $attempt -eq 1 ] && stamp "$name: first try failed, trying once more (details in $R/$name.log)"
    done
    if [ -s $R/$name/results.json ]; then
        stamp "$name done in $(( ($(date +%s) - start) / 60 )) min"
    else
        stamp "$name FAILED - last lines of $R/$name.log:"; tail -8 $R/$name.log; return 1
    fi
}
lane_a() { if has base; then run base --model "$BASE_MODEL"; fi; }
lane_b() { if has grpo; then run grpo --model "$BASE_MODEL" --adapter $ADAPTER; fi; }
if [ "$GA" != "$GB" ]; then
    ( export CUDA_VISIBLE_DEVICES=$GA; lane_a ) > reference_27b.gpu$GA.log 2>&1 &
    ( export CUDA_VISIBLE_DEVICES=$GB; lane_b ) > reference_27b.gpu$GB.log 2>&1 &
    wait
    cat reference_27b.gpu$GA.log reference_27b.gpu$GB.log
else
    ( export CUDA_VISIBLE_DEVICES=$GA; lane_a; lane_b )
fi
if has theirs; then
    ( export CUDA_VISIBLE_DEVICES=$GA; run theirs --model "$THEIRS" ) || echo "NOTE: the comparison model did not run (gated: accept its terms on the Hub and log in)."
fi

for name in base grpo theirs; do
    [ -s $R/$name/results.json ] || continue
    echo; echo "== $name"
    python -m evaluation.reference --out $R/$name --report 2>/dev/null | grep -a -v -i "warn"
done
for name in grpo theirs; do
    [ -s $R/base/results.json ] && [ -s $R/$name/results.json ] || continue
    echo; echo "== $name against base"
    python -m evaluation.reference --compare $R/base $R/$name 2>&1 | grep -a -v -i "warn" | tee $R/$name.compare.txt
done
tar czf results_27b_reference.tar.gz $R bench_reference/questions.jsonl bench_reference/manifest.json reference_27b*.log 2>/dev/null
stamp "ALL DONE - send back results_27b_reference.tar.gz ($(du -h results_27b_reference.tar.gz | cut -f1))"
