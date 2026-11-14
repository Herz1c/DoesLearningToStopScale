#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU machine, then a self-check. Run from anywhere:
#     bash chains/setup.sh
# Needs: Python 3.10-3.12, an NVIDIA GPU with a working driver, internet (PyPI and Hugging Face).
# Creates .venv/ in the repository and installs requirements.txt. Nothing outside the repository
# is changed, except that bubblewrap is installed with apt when it is missing and apt is usable.
cd "$(dirname "$0")/.." || exit 1
ok() { echo "  OK    $*"; }
bad() { echo "  FAIL  $*"; FAILED=1; }

echo "== GPU"
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then ok "GPU visible"; else bad "nvidia-smi does not work"; fi

echo "== Python environment"
if [ ! -x .venv/bin/python ]; then
    PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
    "$PY" -m venv .venv || { bad "could not create .venv with $PY (apt install python3-venv?)"; exit 1; }
fi
source .venv/bin/activate
if [ -z "$SKIP_INSTALL" ]; then        # SKIP_INSTALL=1: use the environment as it is
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt || bad "pip install failed"
fi
python - <<'PY' || FAILED=1
import torch, vllm, transformers, peft
print(f"  torch {torch.__version__}, vllm {vllm.__version__}, transformers {transformers.__version__}, peft {peft.__version__}")
assert torch.cuda.is_available(), "torch does not see the GPU"
print(f"  OK    torch sees {torch.cuda.get_device_name(0)}, bf16 {torch.cuda.is_bf16_supported()}")
PY

echo "== Sandbox for model-written code (bubblewrap)"
command -v bwrap >/dev/null || { command -v apt-get >/dev/null && (sudo -n apt-get install -y bubblewrap || apt-get install -y bubblewrap) >/dev/null 2>&1; }
if command -v bwrap >/dev/null; then
    python - <<'PY' && ok "sandbox runs programs" || bad "bwrap is installed but cannot start a sandbox here (unprivileged container?). The code-domain runs will fail; the other domains are not affected. Fix: run the container with --privileged or --security-opt seccomp=unconfined --cap-add SYS_ADMIN, or use a VM / bare metal."
import sys
sys.path.insert(0, ".")
from stopping.code_tasks import run_programs
good = ("def add(a, b):\n    return a + b\n", "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
wrong = ("def add(a, b):\n    return a - b\n", good[1])
verdicts = run_programs([good, wrong])
assert [v["passed"] for v in verdicts] == [True, False], verdicts
PY
else
    bad "bubblewrap (bwrap) is not installed: apt install bubblewrap"
fi

echo "== Data"
for f in pool4b_math pool4b_code pool4b_tool pool4b_constraint; do
    if [ -s "data/$f.jsonl" ]; then ok "data/$f.jsonl ($(wc -l < "data/$f.jsonl") rows)"; else bad "data/$f.jsonl missing (unpack the data archive into the repository)"; fi
done

echo
if [ -z "$FAILED" ]; then echo "SETUP OK"; else echo "SETUP INCOMPLETE - see FAIL lines above"; exit 1; fi
