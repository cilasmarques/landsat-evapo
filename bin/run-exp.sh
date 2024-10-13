#!/bin/bash

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

OUTPUT_DATA_PATH=./output

./main "$@" &

PID=$!

sh ./scripts/collect-cpu-usage.sh $PID > $OUTPUT_DATA_PATH/cpu.csv &
sh ./scripts/collect-memory-usage.sh $PID > $OUTPUT_DATA_PATH/mem.csv &
sh ./scripts/collect-disk-usage.sh $PID > $OUTPUT_DATA_PATH/disk.csv &
sh ./scripts/collect-gpu-usage.sh $PID > $OUTPUT_DATA_PATH/gpu.csv &
sh ./scripts/collect-gpu-memory-usage.sh $PID > $OUTPUT_DATA_PATH/mem-gpu.csv &

wait $PID

kill $(pidof -s collect-cpu-usage.sh)
kill $(pidof -s collect-memory-usage.sh)
kill $(pidof -s collect-disk-usage.sh)
kill $(pidof -s collect-gpu-usage.sh)
kill $(pidof -s collect-gpu-memory-usage.sh)

exit 0