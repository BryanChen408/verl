#!/usr/bin/env bash
# Host/NPU memory time-series monitor for OOM debugging.
#
# Two parallel loops:
#   1. node-level:  /tmp/mem_node.log     - 1s tick, free + npu-smi
#   2. process-agg: /tmp/mem_procs.log    - 2s tick, RSS per process category
#
# Pair with the in-Python `[MEM ...]` and `[OPT-CHECK ...]` lines emitted by
# verl/utils/debug/mem_probe.py to align node/process state with the exact
# training phase.
#
# Usage:
#     bash scripts/hostoom_monitor.sh start    # spawn both loops, print PIDs
#     bash scripts/hostoom_monitor.sh stop     # kill any running loops
#     bash scripts/hostoom_monitor.sh tail     # tail both logs side-by-side

NODE_LOG=${NODE_LOG:-/tmp/mem_node.log}
PROC_LOG=${PROC_LOG:-/tmp/mem_procs.log}
PIDFILE=${PIDFILE:-/tmp/hostoom_monitor.pids}

_start_node_loop() {
    (
        while true; do
            local ts mem npu
            ts=$(date '+%Y-%m-%d %H:%M:%S.%3N')
            mem=$(free -m | awk 'NR==2{printf "used=%dMB avail=%dMB buff_cache=%dMB", $3, $7, $6}')
            # npu-smi may not be on PATH or may be slow; tolerate failure
            npu=$(timeout 2 npu-smi info 2>/dev/null \
                | awk '/^\| [0-9]+ +[0-9]+/ {printf "die%s=%sMB ", $2, $9}' \
                | head -c 200)
            echo "$ts | $mem | NPU: ${npu:-N/A}"
            sleep 1
        done
    ) > "$NODE_LOG" 2>&1 &
    echo $!
}

_start_proc_loop() {
    (
        while true; do
            local ts
            ts=$(date '+%H:%M:%S')
            ps -eo pid,rss,comm,args --sort=-rss 2>/dev/null | awk -v ts="$ts" '
                NR > 1 {
                    if ($0 ~ /VLLM::Worker/)                       cat = "vllm_worker"
                    else if ($0 ~ /vLLMHttpServer/)                cat = "vllm_http"
                    else if ($0 ~ /WorkerDict.actor_rollout_ref/)  cat = "train_worker"
                    else if ($0 ~ /AgentLoopWorker/)               cat = "agent_loop"
                    else if ($0 ~ /RewardLoopWorker/)              cat = "reward_loop"
                    else if ($0 ~ /TaskRunner/)                    cat = "task_runner"
                    else if ($0 ~ /raylet/)                        cat = "raylet"
                    else if ($0 ~ /plasma_store/)                  cat = "plasma"
                    else if ($0 ~ /python.*default_worker/)        cat = "ray_default"
                    else                                           cat = ""
                    if (cat != "") {
                        agg[cat] += $2
                        count[cat]++
                    }
                }
                END {
                    for (c in agg) printf "%s %s: count=%d total=%dMB\n", ts, c, count[c], agg[c]/1024
                }
            ' | sort
            echo "---"
            sleep 2
        done
    ) > "$PROC_LOG" 2>&1 &
    echo $!
}

case "${1:-start}" in
    start)
        if [[ -f "$PIDFILE" ]]; then
            echo "[hostoom_monitor] $PIDFILE exists; stop first or delete it." >&2
            exit 1
        fi
        : > "$NODE_LOG"
        : > "$PROC_LOG"
        node_pid=$(_start_node_loop)
        proc_pid=$(_start_proc_loop)
        echo "$node_pid $proc_pid" > "$PIDFILE"
        echo "[hostoom_monitor] node loop pid=$node_pid -> $NODE_LOG"
        echo "[hostoom_monitor] proc loop pid=$proc_pid -> $PROC_LOG"
        ;;
    stop)
        if [[ ! -f "$PIDFILE" ]]; then
            echo "[hostoom_monitor] no PID file at $PIDFILE; nothing to stop." >&2
            exit 0
        fi
        for pid in $(cat "$PIDFILE"); do
            if kill "$pid" 2>/dev/null; then
                echo "[hostoom_monitor] killed pid=$pid"
            fi
        done
        rm -f "$PIDFILE"
        ;;
    tail)
        echo "=== $NODE_LOG ==="
        tail -n 30 "$NODE_LOG" 2>/dev/null
        echo
        echo "=== $PROC_LOG ==="
        tail -n 60 "$PROC_LOG" 2>/dev/null
        ;;
    *)
        echo "Usage: $0 {start|stop|tail}" >&2
        exit 1
        ;;
esac
