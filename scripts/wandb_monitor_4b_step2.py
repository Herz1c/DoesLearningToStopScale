#!/usr/bin/env python3
"""Sidecar W&B monitor for generate_4b.sh. Reads progress; does not touch the run.

    pip install wandb && wandb login --host=https://wandb.tool.kinit.sk
    WANDB_BASE_URL=https://wandb.tool.kinit.sk python scripts/wandb_monitor_4b_step2.py

Run it alongside `bash chains/generate_4b.sh`; it polls runs/4b and generate_4b.log
every 30 s and exits when the results tarball appears. Auth comes from ~/.netrc /
wandb login, never from this file.
"""
import os, re, time, subprocess, json
from pathlib import Path

os.environ.setdefault("WANDB_BASE_URL", "https://wandb.tool.kinit.sk")
import wandb

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "4b"
LOG = REPO / "generate_4b.log"
POOLS = {"math": 1500, "code": 1500, "tool": 1500, "constraint": 1600}
ROLLOUT_TARGET = 4 * sum(POOLS.values())          # 24 400 rollouts
HELDOUT_TARGET = 164 * 4 + 500 * 2 + 313 * 4      # 2 908 records (MATH-500: 313 whole-number answers)
STAGES = ["rollouts", "base", "heldout"]

run = wandb.init(project="qwen4b-step2", name="step2-" + time.strftime("%Y%m%d-%H%M%S"),
                 config={"model": "Qwen/Qwen3-4B", "pools": POOLS, "samples_per_question": 4,
                         "rollout_target": ROLLOUT_TARGET, "heldout_target": HELDOUT_TARGET},
                 resume="allow")
wandb.define_metric("rollouts_total", step_metric="tick")
wandb.define_metric("base_records", step_metric="tick")
wandb.define_metric("heldout_records", step_metric="tick")
wandb.define_metric("stages_done", step_metric="tick")

def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(",")
        return dict(gpu_util=float(o[0]), gpu_mem_mb=float(o[1]), gpu_temp_c=float(o[2]), gpu_power_w=float(o[3]))
    except Exception:
        return {}

def lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def log_text():
    try:
        return LOG.read_text(errors="replace")
    except Exception:
        return ""

tick = 0
done_seen = set()
base_results_seen = False
while True:
    text = log_text()
    c = {p: lines(OUT / f"rollouts_{p}" / "rollouts.jsonl") for p in POOLS}
    total = sum(c.values())
    base_n = lines(OUT / "base" / "records.jsonl")
    held_n = lines(OUT / "heldout_base" / "records.jsonl")
    base_done = (OUT / "base" / "results.json").exists()
    held_done = held_n >= HELDOUT_TARGET
    # base total: questions of the selected benchmarks x samples, from the pinned question file
    base_target = None
    try:
        import sys
        sys.path.insert(0, str(REPO))
        from evaluation import protocol, sources
        cfg = json.loads((OUT / "base" / "run_config.json").read_text())
        wanted = {d for b in cfg["benchmarks"] for d in protocol.BENCHMARKS[b]}
        nq = sum(1 for q in sources.read_jsonl(REPO / "bench" / "questions.jsonl") if q["dataset"] in wanted)
        if cfg.get("limit"):
            nq = min(nq, cfg["limit"] * len(wanted))
        base_target = nq * cfg["samples"]
    except Exception:
        pass
    stages_done = sum([all(c[p] >= 4 * n for p, n in POOLS.items()), base_done, held_done])
    newly = []
    for m in re.finditer(r"\] rollouts (\w+) done in (\d+) min", text):
        key = "rollouts_" + m.group(1)
        if key not in done_seen:
            done_seen.add(key); newly.append((key, int(m.group(2))))
    for stage in ("base", "heldout"):
        m = re.search(rf"\] {stage} done in (\d+) min", text)
        if m and stage not in done_seen:
            done_seen.add(stage); newly.append((stage, int(m.group(1))))
    payload = {"tick": tick, "rollouts_total": total, "base_records": base_n, "heldout_records": held_n,
               "stages_done": stages_done,
               "pct_rollouts": round(100 * total / ROLLOUT_TARGET, 1),
               "pct_heldout": round(100 * min(held_n, HELDOUT_TARGET) / HELDOUT_TARGET, 1)}
    if base_target:
        payload["pct_base"] = round(100 * min(base_n, base_target) / base_target, 1)
    payload.update(gpu())
    for p in POOLS:
        payload[f"rollouts/{p}"] = c[p]
    for name, mins in newly:
        payload[f"stage_minutes/{name}"] = mins
    if base_done and not base_results_seen:
        base_results_seen = True
        try:
            results = json.loads((OUT / "base" / "results.json").read_text())["results"]
            for name, r in results.items():
                if isinstance(r, dict) and "accuracy" in r:
                    payload[f"base_accuracy/{name}"] = r["accuracy"]
                    payload[f"base_tokens/{name}"] = r.get("mean_output_tokens", 0)
        except Exception:
            pass
    run.log(payload)
    print(f"[{time.strftime('%H:%M:%S')}] rollouts {total}/{ROLLOUT_TARGET}, base {base_n}, "
          f"heldout {held_n}/{HELDOUT_TARGET}, {stages_done}/3 stages done", flush=True)
    if (REPO / "results_4b_step2.tar.gz").exists() or "ALL DONE" in text:
        run.log({"tick": tick + 1, "rollouts_total": total, "base_records": base_n,
                 "heldout_records": held_n, "stages_done": stages_done, "finished": 1})
        run.summary["finished"] = True
        print("ALL DONE detected — monitor exiting.", flush=True)
        break
    tick += 1
    time.sleep(30)
run.finish()
