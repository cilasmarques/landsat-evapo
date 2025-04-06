#!/bin/bash

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

OUTPUT_DATA_PATH=./output

./main "$@" &
PID=$!

./scripts/gpu-monitor.sh $PID "$OUTPUT_DATA_PATH/gpu_metrics.csv" &
MONITOR_PID_GPU=$!
./scripts/collect-gpu-usage.sh $PID > $OUTPUT_DATA_PATH/gpu.csv &
MONITOR_PID_GPU_L=$!
./scripts/collect-cpu-usage.sh $PID > "$OUTPUT_DATA_PATH/cpu.csv" &
MONITOR_PID_CPU=$!
./scripts/collect-memory-usage.sh $PID > "$OUTPUT_DATA_PATH/mem.csv" &
MONITOR_PID_MEM=$!

wait $PID
EXITCODE=$?
echo "exit code: $EXITCODE"

kill $MONITOR_PID_GPU 2>/dev/null
kill $MONITOR_PID_GPU_L 2>/dev/null
kill $MONITOR_PID_CPU 2>/dev/null
kill $MONITOR_PID_MEM 2>/dev/null

exit $EXITCODE
