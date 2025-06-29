#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PID> # Monitors the specific PID (output to stdout)"
    exit 1
fi

PID_TO_MONITOR="$1"

echo "TIMESTAMP, PID, CPU_USAGE_PERCENTAGE, MEM_USAGE_PERCENTAGE,MEM_USAGE_MB"
while [ -e /proc/$PID_TO_MONITOR ]; do
  if [ -f /proc/$PID_TO_MONITOR/stat ]; then
    stat=$(cat /proc/$PID_TO_MONITOR/stat)
    timestamp=$(date +%s)
    
    # Extract process info
    pid=$(echo $stat | awk '{print $1}')
    cpu_usage=$(ps -p $PID_TO_MONITOR -o %cpu | tail -n 1 | tr -d ' ')
    mem_usage=$(ps -p $PID_TO_MONITOR -o %mem | tail -n 1 | tr -d ' ')
    mem_kb=$(grep 'VmRSS' /proc/$PID_TO_MONITOR/status | awk '{print $2}')
    
    if [ ! -z "$cpu_usage" ] && [ ! -z "$mem_usage" ] && [ ! -z "$mem_kb" ]; then
      echo "$timestamp, $pid, $cpu_usage, $mem_usage, $mem_kb"
    fi
  else
    break
  fi
  # No sleep command to maximize sampling frequency
done

exit 0