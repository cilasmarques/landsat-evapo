#!/bin/bash

# Script to run Nsight Systems profiler and monitor resources

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

# Configuration
OUTPUT_DATA_PATH=./output

# Ensure other processes don't interfere (tries to increase priority)
renice -n -10 $$ >/dev/null 2>&1 || true

# Run the application with Nsight Systems profiling
nsys profile -o ./$OUTPUT_DATA_PATH/nsys ./main "$@" &
APP_PID=$!

# Start resource monitoring
./scripts/collect-cpu-usage.sh $APP_PID > "$OUTPUT_DATA_PATH/cpu_metrics.csv" &
CPU_PID=$!

./scripts/collect-gpu-usage.sh $APP_PID > "$OUTPUT_DATA_PATH/gpu_metrics.csv" &
GPU_PID=$!

# Wait for the main application to finish
wait $APP_PID
EXITCODE=$?

# Terminate monitoring processes
kill $CPU_PID $GPU_PID 2>/dev/null || true

exit $EXITCODE