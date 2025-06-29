#!/bin/bash

# Script to run Nsight Compute profiler and monitor resources

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

# Configuration
OUTPUT_DATA_PATH=./output

# Ensure other processes don't interfere (tries to increase priority)
renice -n -10 $$ >/dev/null 2>&1 || true

# Run the application with NCU profiling
sudo ncu --target-processes all --set full -o ./$OUTPUT_DATA_PATH/ncu ./main "$@" &
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
kill $GPU_PID $CPU_PID 2>/dev/null || true

exit $EXITCODE