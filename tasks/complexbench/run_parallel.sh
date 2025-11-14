#!/bin/bash
# 手动数据并行推理脚本

# ============================================================
# 📝 配置区域 - 根据需要修改以下参数
# ============================================================

# --- 默认值配置 ---
DEFAULT_WORK_DIR="/mnt/shared-storage-user/songdemin/user/guoxu/workspace/ifdecorator/third_party_benchmarks/@inference"
DEFAULT_MODEL_PATH="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
DEFAULT_INPUT_FILE="/mnt/shared-storage-user/songdemin/user/guoxu/workspace/ifdecorator/third_party_benchmarks/ComplexBench/data/data_final.json"
DEFAULT_OUTPUT_FILE="/mnt/shared-storage-user/songdemin/user/guoxu/workspace/ifdecorator/third_party_benchmarks/@inference/tasks/complexbench/results/qwen2.5-7b-instruct.jsonl"

# 解析器配置
DEFAULT_INPUT_PARSER="/mnt/shared-storage-user/songdemin/user/guoxu/workspace/ifdecorator/third_party_benchmarks/@inference/tasks/complexbench/complexbench_input_parser.py"
DEFAULT_INPUT_PARSER_FN="parse_complexbench"
DEFAULT_OUTPUT_PARSER="/mnt/shared-storage-user/songdemin/user/guoxu/workspace/ifdecorator/third_party_benchmarks/@inference/tasks/complexbench/complexbench_output_parser.py"
DEFAULT_OUTPUT_PARSER_FN="parse_complexbench_output"

# 并行和推理参数
DEFAULT_NUM_SHARDS=8
DEFAULT_TENSOR_PARALLEL_SIZE=1
DEFAULT_BATCH_SIZE=128
DEFAULT_TEMPERATURE=0.0
DEFAULT_TOP_P=1.0
DEFAULT_MAX_TOKENS=8192
DEFAULT_TRUST_REMOTE_CODE="--trust_remote_code"
DEFAULT_STARTUP_DELAY=3

# --- 参数解析 ---
# 显示帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

ComplexBench 手动数据并行推理脚本

选项:
  --model_path PATH           模型路径 (默认: $DEFAULT_MODEL_PATH)
  --input_file PATH           输入数据文件 (默认: $DEFAULT_INPUT_FILE)
  --output_file PATH          输出文件 (默认: $DEFAULT_OUTPUT_FILE)
  --work_dir PATH             工作目录 (默认: $DEFAULT_WORK_DIR)
  --num_shards N              数据分片数 (默认: $DEFAULT_NUM_SHARDS)
  --tensor_parallel_size N    张量并行大小 (默认: $DEFAULT_TENSOR_PARALLEL_SIZE)
  --batch_size N              批处理大小 (默认: $DEFAULT_BATCH_SIZE)
  --temperature F             采样温度 (默认: $DEFAULT_TEMPERATURE)
  --top_p F                   nucleus sampling 参数 (默认: $DEFAULT_TOP_P)
  --max_tokens N              最大生成 token 数 (默认: $DEFAULT_MAX_TOKENS)
  --trust_remote_code BOOL    是否信任远程代码 true/false (默认: true)
  --startup_delay N           分片启动间隔秒数 (默认: $DEFAULT_STARTUP_DELAY)
  --input_parser PATH         自定义输入解析器路径
  --input_parser_fn NAME      自定义输入解析函数名
  --output_parser PATH        自定义输出解析器路径
  --output_parser_fn NAME     自定义输出解析函数名
  -h, --help                  显示此帮助信息

示例:
  $0 --model_path /path/to/model --output_file /path/to/output.jsonl
EOF
    exit 0
}

# 初始化变量为默认值
WORK_DIR="$DEFAULT_WORK_DIR"
MODEL_PATH="$DEFAULT_MODEL_PATH"
INPUT_FILE="$DEFAULT_INPUT_FILE"
OUTPUT_FILE="$DEFAULT_OUTPUT_FILE"
NUM_SHARDS="$DEFAULT_NUM_SHARDS"
TENSOR_PARALLEL_SIZE="$DEFAULT_TENSOR_PARALLEL_SIZE"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
TEMPERATURE="$DEFAULT_TEMPERATURE"
TOP_P="$DEFAULT_TOP_P"
MAX_TOKENS="$DEFAULT_MAX_TOKENS"
TRUST_REMOTE_CODE="$DEFAULT_TRUST_REMOTE_CODE"
STARTUP_DELAY="$DEFAULT_STARTUP_DELAY"
INPUT_PARSER="$DEFAULT_INPUT_PARSER"
INPUT_PARSER_FN="$DEFAULT_INPUT_PARSER_FN"
OUTPUT_PARSER="$DEFAULT_OUTPUT_PARSER"
OUTPUT_PARSER_FN="$DEFAULT_OUTPUT_PARSER_FN"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --work_dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --num_shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --tensor_parallel_size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --trust_remote_code)
            if [ "$2" = "true" ]; then
                TRUST_REMOTE_CODE="--trust_remote_code"
            else
                TRUST_REMOTE_CODE=""
            fi
            shift 2
            ;;
        --startup_delay)
            STARTUP_DELAY="$2"
            shift 2
            ;;
        --input_parser)
            INPUT_PARSER="$2"
            shift 2
            ;;
        --input_parser_fn)
            INPUT_PARSER_FN="$2"
            shift 2
            ;;
        --output_parser)
            OUTPUT_PARSER="$2"
            shift 2
            ;;
        --output_parser_fn)
            OUTPUT_PARSER_FN="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# --- 临时工作路径（每次运行创建唯一目录，避免多任务冲突） ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RANDOM_SUFFIX=$RANDOM
TEMP_WORK_DIR="${OUTPUT_FILE%.jsonl}_temp_${TIMESTAMP}_${RANDOM_SUFFIX}"

# --- 日志路径（放在临时目录下，确保完全隔离） ---
LOG_DIR="$TEMP_WORK_DIR/logs"

# ============================================================
# 以下为脚本执行逻辑，一般无需修改
# ============================================================

echo "=================================="
echo "ComplexBench 手动数据并行推理"
echo "=================================="
echo "模型路径: $MODEL_PATH"
echo "输入数据: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "临时工作目录: $TEMP_WORK_DIR"
echo "分片数: $NUM_SHARDS"
echo "张量并行: $TENSOR_PARALLEL_SIZE"
echo "总 GPU 数: $((NUM_SHARDS * TENSOR_PARALLEL_SIZE))"
echo "=================================="

cd $WORK_DIR

# 创建必要的目录
mkdir -p "$TEMP_WORK_DIR"
mkdir -p "$LOG_DIR"
echo "✓ 创建临时工作目录: $TEMP_WORK_DIR"
echo "✓ 创建日志目录: $LOG_DIR"

# 定义临时输出文件（使用临时目录）
TEMP_OUTPUT_FILE="$TEMP_WORK_DIR/output.jsonl"

# 启动所有分片
for ((i=0; i<$NUM_SHARDS; i++)); do
    echo ""
    echo "🚀 启动分片 $i/$NUM_SHARDS..."
    
    python inference.py \
        --model_name_or_path "$MODEL_PATH" \
        --input_file "$INPUT_FILE" \
        --output_file "$TEMP_OUTPUT_FILE" \
        --input_parser $INPUT_PARSER \
        --input_parser_fn $INPUT_PARSER_FN \
        --output_parser $OUTPUT_PARSER \
        --output_parser_fn $OUTPUT_PARSER_FN \
        --shard_id $i \
        --num_shards $NUM_SHARDS \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --batch_size $BATCH_SIZE \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_tokens $MAX_TOKENS \
        $TRUST_REMOTE_CODE \
        > $LOG_DIR/shard_${i}.log 2>&1 &
    
    PIDS[$i]=$!
    echo "   进程 ID: ${PIDS[$i]}"
    echo "   日志文件: $LOG_DIR/shard_${i}.log"
    
    # 延迟启动，避免资源竞争
    sleep $STARTUP_DELAY
done

echo ""
echo "=================================="
echo "✅ 所有分片已启动！"
echo "=================================="
echo ""
echo "监控命令："
echo "  tail -f $LOG_DIR/shard_*.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "等待所有任务完成..."
echo ""

# 等待所有进程完成
ALL_SUCCESS=true
for ((i=0; i<$NUM_SHARDS; i++)); do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ 分片 $i 完成"
    else
        echo "✗ 分片 $i 失败 (退出码: $EXIT_CODE)"
        ALL_SUCCESS=false
    fi
done

echo ""
echo "=================================="
echo "📁 分片输出文件："
TOTAL_LINES=0
for ((i=0; i<$NUM_SHARDS; i++)); do
    SHARD_FILE="${TEMP_OUTPUT_FILE%.jsonl}_shard${i}.jsonl"
    if [ -f "$SHARD_FILE" ]; then
        COUNT=$(wc -l < "$SHARD_FILE")
        TOTAL_LINES=$((TOTAL_LINES + COUNT))
        echo "   分片 $i: $COUNT 条"
    else
        echo "   ✗ 警告: 分片 $i 文件不存在: $SHARD_FILE"
    fi
done
echo "   总计: $TOTAL_LINES 条"
echo ""

# 合并所有分片（使用明确的文件列表，避免通配符问题）
echo "🔄 合并所有分片..."
> "$OUTPUT_FILE"  # 清空或创建输出文件
for ((i=0; i<$NUM_SHARDS; i++)); do
    SHARD_FILE="${TEMP_OUTPUT_FILE%.jsonl}_shard${i}.jsonl"
    if [ -f "$SHARD_FILE" ]; then
        cat "$SHARD_FILE" >> "$OUTPUT_FILE"
        echo "   ✓ 已合并分片 $i"
    fi
done

# 验证合并结果
MERGED_COUNT=$(wc -l < "$OUTPUT_FILE")
echo ""
echo "✅ 合并完成！"
echo "   输出文件: $OUTPUT_FILE"
echo "   总行数: $MERGED_COUNT 条"

if [ "$MERGED_COUNT" -eq "$TOTAL_LINES" ]; then
    echo "   ✓ 行数验证通过"
else
    echo "   ✗ 警告: 合并后行数 ($MERGED_COUNT) 与预期 ($TOTAL_LINES) 不一致"
fi

# 清理临时文件（仅在所有任务成功时）
echo ""
if [ "$ALL_SUCCESS" = true ]; then
    echo "🧹 清理临时文件..."
    rm -rf "$TEMP_WORK_DIR"
    echo "   ✓ 已删除临时目录: $TEMP_WORK_DIR"
else
    echo "⚠️  检测到失败任务，保留日志以供调试："
    echo "   临时目录: $TEMP_WORK_DIR"
    echo "   日志目录: $LOG_DIR"
    echo ""
    echo "查看日志命令："
    echo "   cat $LOG_DIR/shard_*.log"
fi

echo "=================================="

