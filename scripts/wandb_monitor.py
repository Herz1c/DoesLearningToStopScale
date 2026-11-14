#!/usr/bin/env python3
"""Sidecar W&B monitor for calibrate_4b.sh. Reads progress; does not touch the run.

    pip install wandb && wandb login --host=https://wandb.tool.kinit.sk
    WANDB_BASE_URL=https://wandb.tool.kinit.sk python scripts/wandb_monitor.py

Run it alongside `bash chains/calibrate_4b.sh`; it polls runs/cal4b and the log every
30 s and exits when the results tarball appears. Auth comes from ~/.netrc / wandb login,
never from this file.
"""
import os, re, time, subprocess, json
from pathlib import Path

os.environ.setdefault("WANDB_BASE_URL", "https://wandb.tool.kinit.sk")
import wandb

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "cal4b"
LOG = REPO / "calibrate_4b.log"
TIERS = ["math_t0","math_t1","math_t2","code_t0","code_t1","code_t2",
         "tool_t0","tool_t1","tool_t2","constraint_t0","constraint_t1","constraint_t2"]
TARGET = 800  # 200 questions x 4 samples

run = wandb.init(project="qwen4b-calibration", name="cal4b-"+time.strftime("%Y%m%d-%H%M%S"),
                 config={"model":"Qwen/Qwen3-4B","tiers":len(TIERS),"questions_per_tier":200,
                         "samples_per_question":4}, resume="allow")
wandb.define_metric("rollouts_total", step_metric="tick")
wandb.define_metric("tiers_done", step_metric="tick")

def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(",")
        return dict(gpu_util=float(o[0]), gpu_mem_mb=float(o[1]), gpu_temp_c=float(o[2]), gpu_power_w=float(o[3]))
    except Exception:
        return {}

def counts():
    d = {}
    for t in TIERS:
        f = OUT / t / "rollouts.jsonl"
        try: d[t] = sum(1 for _ in open(f)) if f.exists() else 0
        except Exception: d[t] = 0
    return d

tick = 0
done_seen = set()
while True:
    c = counts()
    total = sum(c.values())
    done = sum(1 for t in TIERS if c[t] >= TARGET)
    # parse last "done in" lines for newly finished tiers
    newly = []
    if LOG.exists():
        for m in re.finditer(r"\] (\w+_t\d) done in (\d+) min", LOG.read_text()):
            if m.group(1) not in done_seen:
                done_seen.add(m.group(1)); newly.append((m.group(1), int(m.group(2))))
    payload = {"tick":tick, "rollouts_total":total, "tiers_done":done,
               "pct_complete": round(100*total/(TARGET*len(TIERS)),1)}
    payload.update(gpu())
    for t in TIERS: payload[f"rollouts/{t}"] = c[t]
    for name, mins in newly: payload[f"tier_minutes/{name}"] = mins
    run.log(payload)
    print(f"[{time.strftime('%H:%M:%S')}] {total}/{TARGET*len(TIERS)} rollouts, {done}/12 tiers done", flush=True)
    if (REPO / "calibration_4b_results.tar.gz").exists() or "ALL DONE" in (LOG.read_text() if LOG.exists() else ""):
        run.log({"tick":tick+1,"rollouts_total":total,"tiers_done":done,"finished":1})
        run.summary["finished"] = True
        print("ALL DONE detected — monitor exiting.", flush=True)
        break
    tick += 1
    time.sleep(30)
run.finish()
