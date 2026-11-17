# Sourced by the chains. settle: wait until the card is free before a job that starts vLLM.
#
# Two things went wrong without it. vLLM measures the free memory while it starts and stops with "Error in
# memory profiling" when another process releases memory at that moment. And on the 27B machine the engine
# process of a finished job stayed alive with 121 GiB of the card, so the next job ran out of memory
# (2026-11-17; the scripts now stop their engine themselves, evaluation/chat.py: leave).
# settle waits (up to GPU_WAIT seconds, default 600) until the memory in use on the card is below
# GPU_FREE_MIB (default 8000) and no longer changing. If the card stays busy it says which processes hold
# it and returns 1; the caller decides (the chains start the job anyway and report its failure).
settle() {
    local gpu=${CUDA_VISIBLE_DEVICES:-0} limit=${GPU_FREE_MIB:-8000} wait=${GPU_WAIT:-600}
    local last=-1 now same=0 waited=0
    gpu=${gpu%%,*}
    while [ $waited -lt $wait ]; do
        now=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -z "$now" ] && return 0
        # "no longer changing": within 256 MiB of the last reading (a desktop or another small job moves it a little)
        if [ $last -ge 0 ] && [ $((now - last)) -le 256 ] && [ $((last - now)) -le 256 ]; then same=$((same + 1)); else same=0; fi
        [ $same -ge 2 ] && [ "$now" -lt "$limit" ] && return 0
        [ $waited -gt 0 ] && [ $((waited % 60)) -eq 0 ] && echo "waiting for GPU $gpu: $now MiB in use (limit $limit)"
        last=$now; waited=$((waited + 3)); sleep 3
    done
    echo "WARNING: GPU $gpu is still busy after $wait s ($now MiB in use). Processes on it:"
    nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null
    echo "         If they are left-overs of an earlier run, stop them (kill <pid>) and start the chain again."
    return 1
}

# one_chain <lock file>: refuse to run the same chain twice at once (two copies share the cards and the output files).
one_chain() {
    command -v flock > /dev/null || return 0
    exec 9> "$1"
    flock -n 9 || { echo "this chain is already running here (lock $1). Stop it first, or wait for it."; exit 1; }
}
