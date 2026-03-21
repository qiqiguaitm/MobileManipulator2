#!/bin/bash
# start_combined.sh — 启动两个 combined_server 实例 (每 GPU 一个)
#
# Usage:
#   ./start_combined.sh [start|stop|restart]

set -e
cd "$(dirname "$0")"

ACTION=${1:-start}
PORT_GPU0=8090
PORT_GPU1=8091
LOG_DIR="/tmp/combined_server_logs"

export LD_LIBRARY_PATH=/data1/miniconda3/lib/python3.13/site-packages/tensorrt_libs/:/data1/miniconda3/lib/python3.13/site-packages/nvidia/cudnn/lib/

stop_servers() {
    echo "[stop] Killing combined_server instances..."
    pkill -f "combined_server.py --port $PORT_GPU0" 2>/dev/null || true
    pkill -f "combined_server.py --port $PORT_GPU1" 2>/dev/null || true
    sleep 1
    # force kill stragglers
    pkill -9 -f "combined_server.py --port $PORT_GPU0" 2>/dev/null || true
    pkill -9 -f "combined_server.py --port $PORT_GPU1" 2>/dev/null || true
    echo "[stop] Done."
}

start_servers() {
    # Check if already running
    if pgrep -f "combined_server.py --port $PORT_GPU0" > /dev/null 2>&1; then
        echo "GPU0 server already running on port $PORT_GPU0. Use restart."
        return 1
    fi

    mkdir -p "$LOG_DIR"

    echo "[start] Launching GPU0 on port $PORT_GPU0 ..."
    CUDA_VISIBLE_DEVICES=0 nohup /data1/miniconda3/bin/python3 -u combined_server.py \
        --port "$PORT_GPU0" --gpu 0 \
        > "$LOG_DIR/gpu0.log" 2>&1 &
    PID0=$!
    echo "  PID=$PID0, log=$LOG_DIR/gpu0.log"

    echo "[start] Launching GPU1 on port $PORT_GPU1 ..."
    CUDA_VISIBLE_DEVICES=1 nohup /data1/miniconda3/bin/python3 -u combined_server.py \
        --port "$PORT_GPU1" --gpu 0 \
        > "$LOG_DIR/gpu1.log" 2>&1 &
    PID1=$!
    echo "  PID=$PID1, log=$LOG_DIR/gpu1.log"

    echo ""
    echo "[start] Waiting for servers to load models (~30s)..."
    echo "  Health check:"
    echo "    curl http://localhost:$PORT_GPU0/api/health"
    echo "    curl http://localhost:$PORT_GPU1/api/health"
    echo "  Tail logs:"
    echo "    tail -f $LOG_DIR/gpu0.log"
    echo "    tail -f $LOG_DIR/gpu1.log"
}

case "$ACTION" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    restart)
        stop_servers
        sleep 2
        start_servers
        ;;
    *)
        echo "Usage: $0 [start|stop|restart]"
        exit 1
        ;;
esac
