#!/usr/bin/env bash
# One-hour probe of Qwen3.8-27B on one large card, before the study for that model is planned.
# It measures nothing scientific; it answers: does the model load in vLLM and Transformers, what does its
# thinking block look like, how fast is it, does a LoRA step at the recipe's length fit, does vLLM take the adapter.
#
#     nohup bash chains/probe_27b.sh >> probe_27b.log 2>&1 &        # or inside tmux
#     tail -f probe_27b.log
#
# Needs: the four data/pool4b_*.jsonl files (already there from the 4B run), about 80 GB of free disk for the
# model download, internet. The model needs newer libraries than the 4B run, so this creates its own
# environment in .venv27/ and leaves .venv/ alone.
# A part that fails does not stop the others: every part writes its own report and the chain packs whatever exists
# into results_27b_probe.tar.gz - send that back.
# Optional environment: BASE_MODEL (default Qwen/Qwen3.8-27B), VENV (default .venv27), SKIP_INSTALL=1,
#   PARTS="template generate train lora", PER_POOL=16, TEXT_ONLY=1, LENGTHS=1024,2048,4096,7400.
cd "$(dirname "$0")/.." || exit 1
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B} PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
VENV=${VENV:-.venv27}
PARTS=${PARTS:-template generate train lora}
PER_POOL=${PER_POOL:-16}
LENGTHS=${LENGTHS:-1024,2048,4096,7400}
[ "${TEXT_ONLY:-1}" = 1 ] && TEXT="--text-only" || TEXT=""
R=runs/27b_probe
mkdir -p $R
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
has() { case " $PARTS " in *" $1 "*) return 0;; esac; return 1; }

stamp "probe start: model $BASE_MODEL, parts: $PARTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "code at commit $(git rev-parse --short HEAD 2>/dev/null)"; git status --short 2>/dev/null | head -5
df -h . | tail -1
for pool in math code tool constraint; do [ -s data/pool4b_$pool.jsonl ] || echo "WARNING: data/pool4b_$pool.jsonl is missing"; done

if [ "${SKIP_INSTALL:-0}" != 1 ]; then
    stamp "environment $VENV start"
    {
        [ -d "$VENV" ] || python3 -m venv "$VENV"
        source "$VENV/bin/activate"
        pip install -q -U pip
        pip install -q -U vllm
        pip install -q -U "transformers>=5.8.0" peft accelerate
    } > $R/install.log 2>&1
    stamp "environment done (details in $R/install.log)"
fi
source "$VENV/bin/activate" || { stamp "no environment in $VENV"; exit 1; }
pip list 2>/dev/null | grep -i -E "^(torch|transformers|vllm|peft|accelerate) " | tee $R/versions.txt

# part <label> <report name> <command...>
part() {
    local label=$1 name=$2; shift 2
    stamp "$label start"
    local start; start=$(date +%s)
    "$@" > $R/$name.log 2>&1 < /dev/null
    if [ -s $R/$name.json ] && ! grep -q '"failed"' $R/$name.json; then
        stamp "$label done in $(( ($(date +%s) - start) / 60 )) min"
    else
        stamp "$label FAILED - last lines of $R/$name.log:"; tail -6 $R/$name.log
    fi
}

has template && part "chat template" template python scripts/probe_model.py template --out $R
if has generate; then
    part "vLLM generation" generate python scripts/probe_model.py generate --out $R --per-pool $PER_POOL $TEXT
    if [ -n "$TEXT" ] && { [ ! -s $R/generate.json ] || grep -q '"failed"' $R/generate.json; }; then
        mv $R/generate.log $R/generate_text_only.log 2>/dev/null; mv $R/generate.json $R/generate_text_only.json 2>/dev/null
        TEXT=""
        part "vLLM generation, second try without the text-only option" generate \
            python scripts/probe_model.py generate --out $R --per-pool $PER_POOL
    fi
fi
has train && part "LoRA training steps" train python scripts/probe_model.py train --out $R --lengths $LENGTHS
if has lora; then
    if [ -s $R/adapter/adapter_config.json ]; then
        part "vLLM with the adapter on 45% of the card" lora python scripts/probe_model.py generate --out $R \
            --name lora --adapter $R/adapter --per-pool 4 --samples 2 --memory 0.45 $TEXT
    else
        stamp "vLLM with the adapter: skipped, the training part saved no adapter"
    fi
fi

echo; echo "== summary"
python - <<'EOF'
import json, pathlib
root = pathlib.Path("runs/27b_probe")
def load(name):
    path = root / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None
t, g, tr, lo = (load(n) for n in ("template", "generate", "train", "lora"))
if t:
    print("template   :", "FAILED" if "failed" in t else
          f"<think> in prompt {[p['think_in_prompt'] for p in t['prompts'].values()]}, tokenize=True returns {t.get('tokenize_true_returns')}")
if g:
    print("generation :", "FAILED" if "failed" in g else
          f"{g['tokens_per_second']} tokens/s, load {g['load_seconds']} s, " +
          ", ".join(f"{k} {v['mean_tokens']} tok, capped {v['capped']}/{v['responses']}, knows {v['knows']}" for k, v in g["by_task"].items()))
if tr:
    print("training   :", "FAILED" if "failed" in tr or "class" not in tr else
          f"{tr['class']}, weights {tr['weights_gib']} GiB, targets missing {tr['recipe_targets_missing']}, " +
          ", ".join(f"length {s['length']}: " + (f"{s['seconds'][-1]} s, {s['peak_gib']} GiB" if "seconds" in s else "does not fit") for s in tr.get("steps", [])))
if lo:
    print("adapter    :", "FAILED" if "failed" in lo else f"vLLM took the adapter at {lo['memory_share']} of the card, {lo['tokens_per_second']} tokens/s")
EOF
tar czf results_27b_probe.tar.gz --exclude="adapter_model.safetensors" $R probe_27b.log 2>/dev/null
ls -la results_27b_probe.tar.gz
stamp "PROBE DONE - send back results_27b_probe.tar.gz"
