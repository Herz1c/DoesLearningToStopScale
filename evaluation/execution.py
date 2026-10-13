"""Executes model-written MBPP+ programs in a sandbox, in parallel.

    verdicts = run_programs([(key, "Mbpp/2", program), ...], evalplus_root, mbpp_file, workers=8)
    # {key: {"base": "pass", "plus": "fail"}} or {key: {"error": "..."}}

Sandbox (default "bwrap", bubblewrap): no network, no IPC, a read-only view of the system with
/home and /mnt hidden, only the Python environment, EvalPlus and the MBPP+ file mounted back, a
private /tmp, and prlimit caps of 4 GB address space and 120 CPU-seconds per worker process.
"rlimit" keeps only the resource caps (for machines without user namespaces) and says so.

Programs of one task go to the same worker, so the task's canonical solution runs once.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

WORKER = Path(__file__).with_name("mbpp_worker.py")
MEMORY = 4 * 1024 ** 3
CPU_SECONDS = 120
TIMEOUT = 3 * 3600          # per worker; a stuck worker loses only its own programs


def bwrap_works():
    try:
        return subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/", "/", "true"],
                              capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_programs(items, evalplus_root, mbpp_file, workers=8, sandbox="bwrap"):
    if not items:
        return {}
    if sandbox == "bwrap" and not bwrap_works():
        raise RuntimeError("bubblewrap cannot create a namespace here; use --sandbox rlimit knowingly")
    if sandbox == "rlimit":
        print("WARNING: MBPP+ programs run with resource limits only, no isolation", file=sys.stderr)
    by_task = {}
    for item in items:
        by_task.setdefault(item[1], []).append(item)
    shares = [[] for _ in range(workers)]
    for group in sorted(by_task.values(), key=len, reverse=True):
        min(shares, key=len).extend(group)
    command = worker_command(sandbox, Path(evalplus_root).resolve(), Path(mbpp_file).resolve())
    verdicts = {}
    with ThreadPoolExecutor(len(shares)) as pool:
        for part in pool.map(lambda share: run_share(command, sandbox, share), [s for s in shares if s]):
            verdicts.update(part)
    return verdicts


def run_share(command, sandbox, share):
    payload = "".join(json.dumps({"key": key, "task": task, "program": program}) + "\n"
                      for key, task, program in share)
    try:
        done = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=TIMEOUT,
                              preexec_fn=limit_resources if sandbox == "rlimit" else None)
        answered = {}
        for line in done.stdout.splitlines():
            if line.strip():
                verdict = json.loads(line)
                answered[verdict.pop("key")] = verdict
        failure = f"worker exited {done.returncode}: {done.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        answered, failure = {}, "worker timed out"
    return {key: answered.get(key, {"error": failure}) for key, _, _ in share}


def limit_resources():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))


def worker_command(sandbox, evalplus_root, mbpp_file):
    python = Path(sys.prefix) / "bin" / "python"
    plain = [str(python), "-I", str(WORKER), str(evalplus_root), str(mbpp_file)]
    if sandbox != "bwrap":
        return plain
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--ro-bind", "/", "/", "--tmpfs", "/home", "--tmpfs", "/mnt", "--tmpfs", "/root",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in mounts(python, evalplus_root, mbpp_file):
        command += ["--ro-bind", str(path), str(path)]
    command += ["--chdir", "/tmp", "--clearenv", "--setenv", "HOME", "/tmp",
                "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONHASHSEED", "0",
                "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                "prlimit", f"--as={MEMORY}", f"--cpu={CPU_SECONDS}", "--"] + plain
    return command


def mounts(python, evalplus_root, mbpp_file):
    """What the worker needs back under the hidden directories: the environment (and every prefix
    its python symlink chain passes through, as uv-managed interpreters live in the home directory),
    the base interpreter, EvalPlus, the data file and the worker script."""
    paths = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), evalplus_root, mbpp_file,
             WORKER.resolve()]
    link = python
    while link.is_symlink():
        link = Path(os.path.normpath(link.parent / os.readlink(link)))
        paths.append(link.parent.parent)
    hidden = (Path("/home"), Path("/mnt"), Path("/root"), Path("/tmp"))
    keep = []
    for path in dict.fromkeys(paths):
        if path.exists() and any(h == path or h in path.parents for h in hidden):
            keep.append(path)
    return keep
