#!/usr/bin/env python3
"""Sidecar W&B monitor for train_4b.sh. Reads progress; does not touch the run.

    pip install wandb && wandb login --host=https://wandb.tool.kinit.sk
    WANDB_BASE_URL=https://wandb.tool.kinit.sk python scripts/wandb_monitor_4b_step3.py

Run it alongside `bash chains/train_4b.sh`; it polls runs/4b, checkpoints/4b and
train_4b.log every 30 s and exits when the results tarball appears (ALL DONE or
STOPPED). Auth comes from ~/.netrc / wandb login, never from this file.
"""
import os, re, time, subprocess, json
from pathlib import Path

os.environ.setdefault("WANDB_BASE_URL", "https://wandb.tool.kinit.sk")
import wandb

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "4b"
CKPT = REPO / "checkpoints" / "4b"
LOG = REPO / "train_4b.log"
HELDOUT_TARGET = 164 * 4 + 500 * 2 + 313 * 4      # 2 908 records (MATH-500: 313 whole-number answers)
EVAL_RUNS = ["grpo_full1", "grpo_full2", "mixG_full"]
HELDOUT_RUNS = ["grpo", "mixG"]
STOP_DOMAINS = ["code", "tool", "constraint"]

run = wandb.init(project="qwen4b-step3", entity="mrtineu",
                 name="step3-" + time.strftime("%Y%m%d-%H%M%S"),
                 config={"model": os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B"),
                         "grpo_steps": 50, "grpo_prompts": 24, "grpo_group": 8,
                         "heldout_target": HELDOUT_TARGET},
                 resume="allow")
wandb.define_metric("tick")
wandb.define_metric("data_proofs_done", step_metric="tick")
wandb.define_metric("stages_done", step_metric="tick")
wandb.define_metric("sft_step", step_metric="tick")
wandb.define_metric("sft_loss", step_metric="sft_step")
wandb.define_metric("grpo_step", step_metric="tick")

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

def read(path):
    try:
        return Path(path).read_text(errors="replace")
    except Exception:
        return ""

def tail(path, n=60000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode(errors="replace")
    except Exception:
        return ""

def pct(a, b):
    return round(100 * min(a, b) / b, 1) if b else None

# data-stage proofs, in chain order (see chains/train_4b.sh)
PROOF_FILES = {
    "regrade_math": OUT / "rollouts_math" / "regraded.txt",
    "stop_points_code": OUT / "stop_points_code" / "train.jsonl",
    "stop_points_tool": OUT / "stop_points_tool" / "train.jsonl",
    "stop_points_constraint": OUT / "stop_points_constraint" / "train.jsonl",
    "verified_code_draft": OUT / "verified_code_draft" / "train.jsonl",
    "give_up_code": OUT / "give_up_code" / "train.jsonl",
    "give_up_math": OUT / "give_up_math" / "train.jsonl",
    "shortest_math": OUT / "sft_shortest.jsonl",
    "mix": OUT / "train_mixG.jsonl",
}

tick = 0
done_seen = set()
grpo_logged = 0
sft_logged = 0
compare_seen = set()
held_compare_seen = set()
results_seen = set()
bench_targets = {}
while True:
    text = read(LOG)
    payload = {"tick": tick}
    payload.update(gpu())

    # ---- data stage
    proofs_done = sum(p.exists() and p.stat().st_size > 0 for p in PROOF_FILES.values())
    payload["data_proofs_done"] = proofs_done
    payload["pct_data"] = pct(proofs_done, len(PROOF_FILES))
    for d in STOP_DOMAINS:
        rounds = [int(m) for m in re.findall(r"^round (\d+):", tail(OUT / f"stop_points_{d}.log"), re.M)]
        if rounds:
            payload[f"stop_rounds/{d}"] = max(rounds)
    mix_n = lines(OUT / "train_mixG.jsonl")
    if mix_n:
        payload["mix_examples"] = mix_n

    # ---- SFT
    sft_text = tail(OUT / "train_mixG.log")
    sft_steps = re.findall(r"^step (\d+)/(\d+)\s+loss ([\d.]+)", sft_text, re.M)
    sft_done = (CKPT / "stoppoint_mixG" / "adapter_model.safetensors").exists()
    if sft_steps:
        step, total, loss = sft_steps[-1]
        step, total, loss = int(step), int(total), float(loss)
        if step > sft_logged:
            sft_logged = step
            payload.update(sft_step=step, sft_loss=loss, sft_total_steps=total, pct_sft=pct(step, total))
    elif sft_done:
        payload["pct_sft"] = 100.0

    # ---- GRPO (smoke + main); log.jsonl is appended across resumes
    grpo_rows = []
    for line in read(CKPT / "grpo" / "log.jsonl").splitlines():
        try:
            grpo_rows.append(json.loads(line))
        except Exception:
            pass
    grpo_done = (CKPT / "grpo" / "adapter_model.safetensors").exists()
    smoke_done = (CKPT / "grpo_smoke" / "adapter_model.safetensors").exists()
    if grpo_rows:
        last = grpo_rows[-1]
        payload["grpo_minutes"] = last.get("minutes", 0)
        for row in grpo_rows[grpo_logged:]:
            g = {"grpo_step": row["step"]}
            for d in ("math", "code", "tool", "instruction"):
                if d in row:
                    g[f"grpo/{d}_acc"] = row[d]["acc"]
                    g[f"grpo/{d}_len"] = row[d]["len"]
                    g[f"grpo/{d}_capped"] = row[d]["capped"]
                    g[f"grpo/{d}_mixed"] = row[d]["mixed"]
            run.log(g)
        grpo_logged = len(grpo_rows)
        payload["pct_grpo"] = pct(last["step"], 50)
    elif grpo_done:
        payload["pct_grpo"] = 100.0

    # ---- benchmark evals (records stream in per batch)
    for name in EVAL_RUNS:
        d = OUT / name
        n = lines(d / "records.jsonl")
        if name not in bench_targets:
            try:
                import sys
                sys.path.insert(0, str(REPO))
                from evaluation import protocol, sources
                cfg = json.loads((d / "run_config.json").read_text())
                wanted = {dm for b in cfg["benchmarks"] for dm in protocol.BENCHMARKS[b]}
                nq = sum(1 for q in sources.read_jsonl(REPO / "bench" / "questions.jsonl") if q["dataset"] in wanted)
                if cfg.get("limit"):
                    nq = min(nq, cfg["limit"] * len(wanted))
                bench_targets[name] = nq * cfg["samples"]
            except Exception:
                pass
        if name in bench_targets:
            payload[f"eval_records/{name}"] = n
            payload[f"pct_eval/{name}"] = pct(n, bench_targets[name])
        m = re.search(r"^(\d+)/(\d+)\s*$", tail(d.with_suffix(".log")), re.M)
        if m:
            payload[f"pct_eval/{name}"] = pct(int(m.group(1)), int(m.group(2)))
        rj = d / "results.json"
        if rj.exists() and name not in results_seen:
            results_seen.add(name)
            try:
                for bname, r in json.loads(rj.read_text())["results"].items():
                    if isinstance(r, dict) and "accuracy" in r:
                        payload[f"accuracy/{name}/{bname}"] = r["accuracy"]
                        payload[f"tokens/{name}/{bname}"] = r.get("mean_output_tokens", 0)
            except Exception:
                pass
        cj = OUT / f"{name}.compare.json"
        if cj.exists() and name not in compare_seen:
            compare_seen.add(name)
            try:
                c = json.loads(cj.read_text())
                avg = c.get("average") or {}
                if avg:
                    payload[f"result/{name}/tokens_saved"] = avg["token_reduction"]
                    payload[f"result/{name}/accuracy_delta_pp"] = 100 * avg["accuracy_delta"]
                    lo, hi = avg["interval"]["accuracy_delta"]
                    payload[f"result/{name}/accuracy_ci_lo_pp"] = 100 * lo
                    payload[f"result/{name}/accuracy_ci_hi_pp"] = 100 * hi
                for bname, b in c.get("benchmarks", {}).items():
                    payload[f"result/{name}/{bname}/tokens_saved"] = 1 - b["token_ratio"]
                    payload[f"result/{name}/{bname}/accuracy_delta_pp"] = 100 * b["accuracy_delta"]
            except Exception:
                pass

    # ---- held-out evals (records only appear when a set finishes)
    for name in HELDOUT_RUNS:
        n = lines(OUT / f"heldout_{name}" / "records.jsonl")
        payload[f"heldout_records/{name}"] = n
        payload[f"pct_heldout/{name}"] = pct(n, HELDOUT_TARGET)
        ct = OUT / f"heldout_{name}.compare.txt"
        if ct.exists() and name not in held_compare_seen:
            held_compare_seen.add(name)
            try:
                ht = ct.read_text(errors="replace")
                for set_name in ("humaneval", "gsm8k_test", "math500"):
                    m = re.search(rf"^{set_name}\s+\(\d+ questions\).*?\n"
                                  rf"\s+tokens saved\s+([+-][\d.]+)%.*?\n"
                                  rf"\s+accuracy change\s+([+-][\d.]+) pp", ht, re.M | re.S)
                    if m:
                        payload[f"result/heldout_{name}/{set_name}/tokens_saved"] = float(m.group(1)) / 100
                        payload[f"result/heldout_{name}/{set_name}/accuracy_delta_pp"] = float(m.group(2))
            except Exception:
                pass

    # ---- stage accounting from the chain log
    stages_done = sum([proofs_done == len(PROOF_FILES), sft_done, smoke_done, grpo_done,
                       (OUT / "grpo_full1" / "results.json").exists(),
                       (OUT / "grpo_full2" / "results.json").exists(),
                       lines(OUT / "heldout_grpo" / "records.jsonl") >= HELDOUT_TARGET,
                       (OUT / "mixG_full" / "results.json").exists(),
                       lines(OUT / "heldout_mixG" / "records.jsonl") >= HELDOUT_TARGET])
    payload["stages_done"] = stages_done
    for m in re.finditer(r"\] (.+?) done in (\d+) min", text):
        key = m.group(1)
        if key not in done_seen:
            done_seen.add(key)
            payload[f"stage_minutes/{key.replace(' ', '_')}"] = int(m.group(2))

    run.log(payload)
    print(f"[{time.strftime('%H:%M:%S')}] data {proofs_done}/{len(PROOF_FILES)}, "
          f"sft {sft_logged}, grpo {grpo_logged}/50, stages {stages_done}/9", flush=True)
    stopped = re.search(r"STOPPED at: (.+)", text)
    if (REPO / "results_4b_step3.tar.gz").exists() or "ALL DONE" in text or stopped:
        run.log({"tick": tick + 1, "stages_done": stages_done, "finished": 1,
                 "stopped": int(bool(stopped))})
        run.summary["finished"] = True
        run.summary["stopped"] = bool(stopped)
        if stopped:
            run.summary["stopped_at"] = stopped.group(1)
            print(f"STOPPED detected: {stopped.group(1)} — monitor exiting.", flush=True)
        else:
            print("ALL DONE detected — monitor exiting.", flush=True)
        break
    tick += 1
    time.sleep(30)
run.finish()
