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

# Todo1: 定义模型列表 格式重要*(model_path|model_name)*
models=(
    "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75|qwen2.5-7b-instruct"
)
# 结果目录
results_dir="/mnt/shared-storage-user/songdemin/user/guoxu/workspace/ifdecorator/third_party_benchmarks/@inference/tasks/complexbench/results"
mkdir -p "$results_dir"

# 批量运行
for model in "${models[@]}"; do
    model_path=$(echo "$model" | cut -d '|' -f 1)
    model_name=$(echo "$model" | cut -d '|' -f 2)
    
    # Todo2: 根据显卡数量和模型设置tensor_parallel_size和num_shards
    if [[ "$model_name" == *"72B"* ]]; then
        extra_args="--tensor_parallel_size 2 --num_shards 1"
        echo "检测到72B模型，使用 tensor_parallel_size=2, num_shards=1"
    else
        extra_args="--tensor_parallel_size 1 --num_shards 2"
    fi
    
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