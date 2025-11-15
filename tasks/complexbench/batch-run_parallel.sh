#!/bin/bash
# 解析命令行参数
dry_run=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry)
            dry_run=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 配置总显卡数
TOTAL_GPUS=2

# Todo1: 定义模型列表 格式重要*(model_path|model_name|tensor_parallel_size)*
models=(
    "/mnt/shared-storage-user/songdemin/user/guoxu/public/hf_hub/models/model--Qwen-Qwen2.5-72B-Instruct|qwen2.5-72B-instruct-origin|2"
)

# 结果目录
results_dir="/mnt/shared-storage-user/songdemin/user/guoxu/workspace/ifdecorator/third_party_benchmarks/@inference/tasks/complexbench/results"
mkdir -p "$results_dir"

# 批量运行
for model in "${models[@]}"; do
    model_path=$(echo "$model" | cut -d '|' -f 1)
    model_name=$(echo "$model" | cut -d '|' -f 2)
    tensor_parallel_size=$(echo "$model" | cut -d '|' -f 3)
    
    # 根据总显卡数和 tensor_parallel_size 计算 num_shards
    num_shards=$((TOTAL_GPUS / tensor_parallel_size))
    total_used_gpus=$((tensor_parallel_size * num_shards))
    
    extra_args="--tensor_parallel_size $tensor_parallel_size --num_shards $num_shards"
    echo "模型: $model_name | TP=$tensor_parallel_size, Shards=$num_shards, 使用GPU数=$total_used_gpus/$TOTAL_GPUS"
    
    if [ "$dry_run" = true ]; then
        echo "=================================="
        echo "dry run: $model_name"
        echo "bash run_parallel.sh --model_path \"$model_path\" --output_file \"$results_dir/${model_name}.jsonl\" $extra_args"
        echo ""
    else
        echo "=================================="
        echo "执行: $model_name"
        echo "bash run_parallel.sh --model_path \"$model_path\" --output_file \"$results_dir/${model_name}.jsonl\" $extra_args"
        bash run_parallel.sh --model_path "$model_path" --output_file "$results_dir/${model_name}.jsonl" $extra_args
        echo "完成: $model_name"
        echo ""
    fi
done

echo "=================================="
echo "✅ 所有模型评测完成！"
echo "结果目录: $results_dir"
echo "=================================="