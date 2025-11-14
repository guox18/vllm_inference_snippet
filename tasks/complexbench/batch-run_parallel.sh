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
# 定义模型列表
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

    if [ "$dry_run" = true ]; then
        echo "dry run:"
        echo "bash run_parallel.sh --model_path \"$model_path\" --output_file \"$results_dir/${model_name}.jsonl\""
    else
        echo "executing:"
        echo "bash run_parallel.sh --model_path \"$model_path\" --output_file \"$results_dir/${model_name}.jsonl\""
        bash run_parallel.sh --model_path "$model_path" --output_file "$results_dir/${model_name}.jsonl"
    fi
done