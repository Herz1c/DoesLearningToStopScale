# Sourced by the chains. settle: wait until the previous process has given its GPU memory back.
# vLLM measures the free memory while it starts and stops with "Error in memory profiling" when another
# process releases memory at that moment - which the process that just ended is still doing.
settle() {
    local gpu=${CUDA_VISIBLE_DEVICES:-0} last=-1 now same=0 tries=0
    gpu=${gpu%%,*}
    while [ $tries -lt 40 ]; do
        now=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -z "$now" ] && return 0
        if [ "$now" = "$last" ]; then same=$((same + 1)); else same=0; fi
        [ $same -ge 2 ] && return 0
        last=$now; tries=$((tries + 1)); sleep 3
    done
}
