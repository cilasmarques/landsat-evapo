#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PID> # Monitors SSD read/write times for the specific PID (output to stdout)"
    exit 1
fi

PID_TO_MONITOR="$1"

echo "TIMESTAMP,PID,READ_TIME_MS,WRITE_TIME_MS,READ_BYTES,WRITE_BYTES,READ_OPERATIONS,WRITE_OPERATIONS,IO_WAIT_TIME_MS,AVG_READ_TIME_MS,AVG_WRITE_TIME_MS"
while [ -e /proc/$PID_TO_MONITOR ]; do
  if [ -f /proc/$PID_TO_MONITOR/stat ]; then
    timestamp=$(date +%s)
    
    # Extract process info
    pid=$(echo $PID_TO_MONITOR)
    
    # Get disk I/O statistics for the process
    read_bytes="N/A"
    write_bytes="N/A"
    read_ops="N/A"
    write_ops="N/A"
    read_time="N/A"
    write_time="N/A"
    io_wait="N/A"
    avg_read_time="N/A"
    avg_write_time="N/A"
    
    if [ -f "/proc/$PID_TO_MONITOR/io" ]; then
        read_bytes=$(grep "rchar" /proc/$PID_TO_MONITOR/io | awk '{print $2}')
        write_bytes=$(grep "wchar" /proc/$PID_TO_MONITOR/io | awk '{print $2}')
        read_ops=$(grep "syscr" /proc/$PID_TO_MONITOR/io | awk '{print $2}')
        write_ops=$(grep "syscw" /proc/$PID_TO_MONITOR/io | awk '{print $2}')
    fi
    
    # Get disk I/O times from /proc/diskstats (more accurate)
    if [ -f "/proc/diskstats" ]; then
        # Read time (field 4) and write time (field 8) in milliseconds
        read_time=$(cat /proc/diskstats | awk '{sum+=$4} END {print sum}')
        write_time=$(cat /proc/diskstats | awk '{sum+=$8} END {print sum}')
        # IO wait time (field 13) in milliseconds
        io_wait=$(cat /proc/diskstats | awk '{sum+=$13} END {print sum}')
        
        # Calculate average times per operation
        if [ ! -z "$read_ops" ] && [ "$read_ops" != "0" ] && [ "$read_ops" != "N/A" ]; then
            avg_read_time=$(echo "scale=3; $read_time / $read_ops" | bc 2>/dev/null)
        fi
        
        if [ ! -z "$write_ops" ] && [ "$write_ops" != "0" ] && [ "$write_ops" != "N/A" ]; then
            avg_write_time=$(echo "scale=3; $write_time / $write_ops" | bc 2>/dev/null)
        fi
    fi
    
    # Alternative method using iostat if available (more detailed)
    if command -v iostat >/dev/null 2>&1; then
        # Get current disk I/O statistics
        iostat_output=$(iostat -x 1 1 2>/dev/null | tail -n +4)
        if [ ! -z "$iostat_output" ]; then
            # Extract average service time
            avg_service_time=$(echo "$iostat_output" | awk '{sum+=$10} END {print sum/NR}')
            if [ ! -z "$avg_service_time" ] && [ "$avg_service_time" != "0" ]; then
                read_time=$avg_service_time
                write_time=$avg_service_time
            fi
        fi
    fi
    
    # Print CSV line
    echo "$timestamp,$pid,$read_time,$write_time,$read_bytes,$write_bytes,$read_ops,$write_ops,$io_wait,$avg_read_time,$avg_write_time"
  else
    break
  fi
  # Minimal sleep for faster sampling
  sleep 0.01  # Reduced from 0.1 to 0.01
done

exit 0 