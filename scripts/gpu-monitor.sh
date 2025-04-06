#!/bin/bash

# GPU Performance Monitor - Simplified Version
# This script monitors various GPU metrics for a specific process

# Check if PID was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <PID> [output_file]"
    exit 1
fi

PID=$1
OUTPUT_FILE=${2:-"gpu_metrics_$PID.csv"}

# Check if PID exists
if ! ps -p $PID > /dev/null; then
    echo "Error: Process with PID $PID does not exist"
    exit 1
fi

# Set numeric format for consistency
export LC_NUMERIC="C"

# Sampling frequency (in seconds)
SAMPLING_RATE=0.5

# Print CSV header
echo "Timestamp,GPU_Util%,Mem_Util%,Mem_Used_MB,Mem_Total_MB,Power_W,Temp_C,PCIe_MB/s" > $OUTPUT_FILE

echo "Starting GPU monitoring for process $PID. Press Ctrl+C to stop."
echo "Results will be saved to $OUTPUT_FILE"

# Function to get GPU metrics
get_gpu_metrics() {
    local pid=$1
    local timestamp=$(date +%s.%3N)
    
    # Basic GPU metrics
    local gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits)
    
    # PCIe statistics (RX + TX)
    local pcie_rx=$(nvidia-smi --query-gpu=pcie.rx_throughput --format=csv,noheader,nounits | sed 's/ MiB\/s//')
    local pcie_tx=$(nvidia-smi --query-gpu=pcie.tx_throughput --format=csv,noheader,nounits | sed 's/ MiB\/s//')
    
    # Calculate total PCIe throughput (if available)
    local pcie_total="N/A"
    if [[ $pcie_rx =~ ^[0-9.]+$ ]] && [[ $pcie_tx =~ ^[0-9.]+$ ]]; then
        pcie_total=$(echo "$pcie_rx + $pcie_tx" | bc)
    fi
    
    # Extract individual values
    local gpu_util=$(echo $gpu_stats | awk -F ', ' '{print $1}')
    local mem_util=$(echo $gpu_stats | awk -F ', ' '{print $2}')
    local mem_used=$(echo $gpu_stats | awk -F ', ' '{print $3}')
    local mem_total=$(echo $gpu_stats | awk -F ', ' '{print $4}')
    local temp=$(echo $gpu_stats | awk -F ', ' '{print $5}')
    local power=$(echo $gpu_stats | awk -F ', ' '{print $6}')
    
    # Save to CSV file
    echo "$timestamp,$gpu_util,$mem_util,$mem_used,$mem_total,$power,$temp,$pcie_total" >> $OUTPUT_FILE
}

# Monitoring loop
while ps -p $PID > /dev/null; do
    get_gpu_metrics $PID
    sleep $SAMPLING_RATE
done

echo "Process $PID has terminated. GPU monitoring finished."
echo "Results saved to $OUTPUT_FILE"

exit 0