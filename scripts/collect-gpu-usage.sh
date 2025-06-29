#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PID> # Monitors the specific PID (output to stdout)"
    exit 1
fi

PID_TO_MONITOR="$1"

# Warm up the GPU to ensure stable readings
nvidia-smi >/dev/null 2>&1
sleep 0.05
nvidia-smi >/dev/null 2>&1

echo "TIMESTAMP,GPU_USAGE_PERCENTAGE,MEM_USAGE_PERCENTAGE,MEM_USAGE_MB,MEM_TOTAL_MB,POWER_W,TEMP_C"
while [ -e /proc/$PID_TO_MONITOR ]; do
    timestamp=$(date +%s)
    
    # Collect GPU metrics
    gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits)
    
    # Extract individual values
    gpu_util=$(echo "$gpu_stats" | awk -F ', ' '{print $1}')
    mem_util=$(echo "$gpu_stats" | awk -F ', ' '{print $2}')
    mem_used=$(echo "$gpu_stats" | awk -F ', ' '{print $3}')
    mem_total=$(echo "$gpu_stats" | awk -F ', ' '{print $4}')
    temp=$(echo "$gpu_stats" | awk -F ', ' '{print $5}')
    power=$(echo "$gpu_stats" | awk -F ', ' '{print $6}')
    
    # Print CSV line
    echo "$timestamp,$gpu_util,$mem_util,$mem_used,$mem_total,$power,$temp"    

    # No sleep command to maximize sampling frequency
done

exit 0