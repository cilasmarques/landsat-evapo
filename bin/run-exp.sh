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

./scripts/collect-cpu-usage.sh $APP_PID > "$OUTPUT_DATA_PATH/cpu_metrics.csv" &
CPU_PID=$!

# Wait for the main application to finish
wait $APP_PID
EXITCODE=$?

# Terminate monitoring processes (they usually end automatically)
kill $CPU_PID 2>/dev/null || true

exit $EXITCODE