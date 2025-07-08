#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PID> # Monitors CPU power consumption for the specific PID (output to stdout)"
    exit 1
fi

PID_TO_MONITOR="$1"

echo "TIMESTAMP,PID,CPU_POWER_W,CPU_FREQ_MHZ,CPU_TEMP_C,PACKAGE_POWER_W,CORES_ACTIVE,CPU_UTIL_PERCENT"
while [ -e /proc/$PID_TO_MONITOR ]; do
  if [ -f /proc/$PID_TO_MONITOR/stat ]; then
    timestamp=$(date +%s)
    
    # Extract process info
    pid=$(echo $PID_TO_MONITOR)
    
    # Get CPU power consumption using multiple methods
    cpu_power="N/A"
    
    # Method 1: Intel RAPL (most accurate) - without internal sleep
    if [ -d "/sys/class/powercap/intel-rapl:0" ]; then
        # Get current energy reading
        energy_current=$(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null)
        if [[ "$energy_current" =~ ^[0-9]+$ ]]; then
            # Convert energy to power estimate (rough calculation)
            cpu_power=$(echo "scale=3; $energy_current / 1000000" | bc 2>/dev/null)
        fi
    fi
    
    # Method 2: Try to get from turbostat if available
    if [ "$cpu_power" = "N/A" ] && command -v turbostat >/dev/null 2>&1; then
        turbostat_output=$(turbostat -q -n 1 2>/dev/null | grep "Core" | awk '{print $4}')
        if [[ "$turbostat_output" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            cpu_power=$turbostat_output
        fi
    fi
    
    # Method 3: Estimate from frequency and utilization (fastest method)
    if [ "$cpu_power" = "N/A" ]; then
        cpu_util=$(ps -p $PID_TO_MONITOR -o %cpu | tail -n 1 | tr -d ' ')
        cpu_freq=$(cat /proc/cpuinfo | grep "cpu MHz" | head -1 | awk '{print $4}')
        # Rough estimation: power ~ frequency * utilization / 1000
        if [[ "$cpu_util" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$cpu_freq" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            cpu_power=$(echo "scale=3; $cpu_freq * $cpu_util / 1000" | bc 2>/dev/null)
        fi
    fi
    
    # Validate cpu_power is numeric
    if [[ ! "$cpu_power" =~ ^[0-9]+\.?[0-9]*$ ]] || [ "$cpu_power" = "N/A" ]; then
        cpu_power="N/A"
    fi
    
    # Get CPU frequency
    cpu_freq=$(cat /proc/cpuinfo | grep "cpu MHz" | head -1 | awk '{print $4}')
    if [[ ! "$cpu_freq" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        cpu_freq="N/A"
    fi
    
    # Get CPU temperature (if available)
    cpu_temp="N/A"
    if [ -f "/sys/class/thermal/thermal_zone0/temp" ]; then
        temp_raw=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
        if [[ "$temp_raw" =~ ^[0-9]+$ ]]; then
            cpu_temp=$(echo "scale=1; $temp_raw / 1000" | bc 2>/dev/null)
        fi
    fi
    
    # Get package power (same as cpu_power for now)
    package_power=$cpu_power
    
    # Count active CPU cores
    cores_active=$(nproc)
    if [[ ! "$cores_active" =~ ^[0-9]+$ ]]; then
        cores_active="N/A"
    fi
    
    # Get CPU utilization for the process
    cpu_util=$(ps -p $PID_TO_MONITOR -o %cpu | tail -n 1 | tr -d ' ')
    if [[ ! "$cpu_util" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        cpu_util="N/A"
    fi
    
    # Print CSV line
    echo "$timestamp,$pid,$cpu_power,$cpu_freq,$cpu_temp,$package_power,$cores_active,$cpu_util"
  else
    break
  fi
  # Minimal sleep for faster sampling
  sleep 0.01  # Reduced from 0.1 to 0.01
done

exit 0 