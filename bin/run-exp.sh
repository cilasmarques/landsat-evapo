#!/bin/bash

# Script to run the application and monitor resources
# Adapted to capture metrics even for extremely fast executions (<1ms)

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

# Configuration
OUTPUT_DATA_PATH=./output

# Ensure other processes don't interfere (tries to increase priority)
renice -n -10 $$ >/dev/null 2>&1 || true

# GPU, CPU and memory monitoring using consistent method for all resources
# Run the application being monitored
./main $* &
APP_PID=$!

# Start resource monitoring
./scripts/collect-gpu-usage.sh $APP_PID > "$OUTPUT_DATA_PATH/gpu_metrics.csv" &
GPU_PID=$!

./scripts/collect-cpu-usage.sh $APP_PID > "$OUTPUT_DATA_PATH/cpu_metrics.csv" &
CPU_PID=$!

./scripts/collect-cpu-power.sh $APP_PID > "$OUTPUT_DATA_PATH/cpu_power_metrics.csv" &
CPU_POWER_PID=$!

./scripts/collect-ssd-io.sh $APP_PID > "$OUTPUT_DATA_PATH/ssd_io_metrics.csv" &
SSD_IO_PID=$!

# Wait for the main application to finish
wait $APP_PID
EXITCODE=$?

# Terminate monitoring processes (they usually end automatically)
kill $GPU_PID $CPU_PID $CPU_POWER_PID $SSD_IO_PID 2>/dev/null || true

exit $EXITCODE