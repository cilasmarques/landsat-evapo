#!/bin/bash

# GPU Performance Monitor para Aplicações Ultra-Rápidas com Precisão de Nanossegundos
# Este script captura métricas de GPU mesmo para aplicações que executam em menos de 1ms

# Verificar se os argumentos foram fornecidos
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Uso: $0 <comando_a_executar> <arquivo_saida> [argumentos_adicionais]"
    echo "Exemplo: $0 './main' gpu_metrics.csv arg1 arg2"
    exit 1
fi

COMMAND_TO_RUN="$1"
OUTPUT_FILE="$2"
shift 2  # Remover os dois primeiros argumentos

# Salvar PID atual para uso em arquivos temporários
SCRIPT_PID=$$

# Criar diretório temporário exclusivo para este processo
TEMP_DIR=$(mktemp -d /tmp/gpu_monitor_${SCRIPT_PID}_XXXXXX)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Configurar formato numérico para consistência
export LC_NUMERIC="C"

# Configurar taxa de amostragem extremamente alta (o intervalo mais curto possível)
SAMPLE_INTERVAL=0.000001  # 1 microssegundo (teórico, na prática depende do sistema)

echo "Iniciando monitoramento de GPU de alta precisão..."

# Criar cabeçalho CSV
echo "Timestamp_ns,GPU_Util%,Mem_Util%,Mem_Used_MB,Mem_Total_MB,Power_W,Temp_C,PCIe_MB/s" > "$OUTPUT_FILE"

# Função para obter métricas GPU com timestamp em nanossegundos
get_gpu_metrics() {
    # Obter timestamp em nanossegundos
    local timestamp=$(date +%s%N)
    
    # Basic GPU metrics
    local gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits)
    
    # Extrair valores individuais - otimizado para evitar chamadas adicionais
    local gpu_util=$(echo "$gpu_stats" | awk -F ', ' '{print $1}')
    local mem_util=$(echo "$gpu_stats" | awk -F ', ' '{print $2}')
    local mem_used=$(echo "$gpu_stats" | awk -F ', ' '{print $3}')
    local mem_total=$(echo "$gpu_stats" | awk -F ', ' '{print $4}')
    local temp=$(echo "$gpu_stats" | awk -F ', ' '{print $5}')
    local power=$(echo "$gpu_stats" | awk -F ', ' '{print $6}')
    
    # Salvar no arquivo CSV diretamente
    echo "$timestamp,$gpu_util,$mem_util,$mem_used,$mem_total,$power,$temp" >> "$OUTPUT_FILE"
}

# 1. Aquecer a GPU para minimizar variações iniciais e cache nvidia-smi
nvidia-smi >/dev/null
sleep 0.05
nvidia-smi >/dev/null

# 2. Coletar linha de base antes da execução
get_gpu_metrics

# 3. Criar um pipe nomeado para sincronização precisa
SYNC_PIPE="$TEMP_DIR/sync.pipe"
mkfifo "$SYNC_PIPE"

# 4. Criar arquivo de controle para sinalizar fim do monitoramento
DONE_FILE="$TEMP_DIR/done"

# 5. Iniciar processo de monitoramento intensivo em background com prioridade alta
(
    # Tentar aumentar prioridade do processo monitor para melhorar amostragem
    renice -n -10 $$ >/dev/null 2>&1 || true
    
    # Aguardar pelo sinal para iniciar monitoramento com timeout reduzido
    read -t 5 signal < "$SYNC_PIPE" || signal="timeout"
    
    if [ "$signal" = "start" ]; then
        # Loop de monitoramento de alta frequência
        while [ ! -f "$DONE_FILE" ]; do
            get_gpu_metrics
            # Mínimo intervalo possível - se falhar no sleep, continua mesmo assim
            sleep $SAMPLE_INTERVAL 2>/dev/null || true
        done
    else
        echo "Erro: Timeout esperando pelo sinal de início" >&2
    fi
) &
MONITOR_PID=$!

# Esperar um pouco para garantir que o monitor esteja pronto
sleep 0.01

# 6. Preparar cache e buffers para execução com precisão de nanossegundos
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# Tentar aumentar prioridade do processo principal
renice -n -20 $$ >/dev/null 2>&1 || true

# 7. Iniciar medição de tempo em nanossegundos - ANTES de iniciar monitoramento
START_NANO=$(date +%s%N)

# 8. Sinalizar início do monitoramento de alta frequência e aguardar confirmação
echo "start" > "$SYNC_PIPE"

# 9. Executar o comando imediatamente após o início do monitoramento
echo "Executando: $COMMAND_TO_RUN $@"
eval "$COMMAND_TO_RUN $@"
COMMAND_EXIT_CODE=$?

# 10. Registrar tempo final em nanossegundos IMEDIATAMENTE após execução
END_NANO=$(date +%s%N)
ELAPSED_NANOS=$((END_NANO - START_NANO))
ELAPSED_SECONDS=$(echo "scale=9; $ELAPSED_NANOS / 1000000000" | bc)

# 11. Sinalizar fim do monitoramento e coletar amostra final
touch "$DONE_FILE"

# Coletar amostra final e garantir que o processo de monitoramento termine
get_gpu_metrics
sleep 0.01
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# 12. Limpar arquivos temporários automaticamente via trap EXIT

echo "Execução concluída com código de saída: $COMMAND_EXIT_CODE"
echo "Tempo de execução: $ELAPSED_SECONDS segundos ($ELAPSED_NANOS nanossegundos)"
echo "Métricas GPU salvas em $OUTPUT_FILE"

exit $COMMAND_EXIT_CODE