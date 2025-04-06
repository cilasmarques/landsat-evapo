#!/bin/bash

# Script para executar o aplicativo e monitorar recursos
# Adaptado para capturar métricas mesmo em execuções extremamente rápidas (<1ms)

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

# Configuração
OUTPUT_DATA_PATH=./output

# Limpar cache do sistema para medições mais precisas
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# Pré-aquecer a GPU (importante para medições de aplicações ultra-rápidas)
nvidia-smi >/dev/null 2>&1
sleep 0.1

# Garantir que outros processos não interfiram (tenta aumentar prioridade)
renice -n -10 $$ >/dev/null 2>&1 || true

# Monitoramento da GPU usando o novo método que inicia o monitoramento ANTES da execução
./scripts/gpu-monitor.sh "./main $*" "$OUTPUT_DATA_PATH/gpu_metrics.csv"
EXITCODE=$?

exit $EXITCODE