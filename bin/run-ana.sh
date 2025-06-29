#!/bin/bash

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

OUTPUT_DATA_PATH=./output

for i in $(seq -f "%02g" 1 1100); do
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

    # Wait for the main application to finish
    wait $APP_PID
    EXITCODE=$?

    # Terminate monitoring processes (they usually end automatically)
    kill $GPU_PID $CPU_PID 2>/dev/null || true

    METHOD=`echo "$@" | grep -oP '(?<=-meth=)[0-9]+'`
    ANALYSIS_OUTPUT_PATH=$OUTPUT_DATA_PATH/kernels-$METHOD
    
    mkdir -p $ANALYSIS_OUTPUT_PATH/experiment${i}
    mv $OUTPUT_DATA_PATH/*.csv $ANALYSIS_OUTPUT_PATH/experiment${i}
    mv $OUTPUT_DATA_PATH/*.txt $ANALYSIS_OUTPUT_PATH/experiment${i}

    sleep 1
done

exit 0
