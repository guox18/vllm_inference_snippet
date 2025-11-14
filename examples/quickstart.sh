#!/bin/bash
# 快速启动脚本 - 自动生成示例文件并运行推理

set -e

echo "=================================="
echo "vLLM 推理框架 - 快速启动"
echo "=================================="

# 创建示例目录
mkdir -p examples/data

echo ""
echo "📝 步骤 1: 生成示例输入数据..."

# 生成示例输入文件
cat > examples/data/sample_input.jsonl << 'EOF'
{"id": 1, "instruction": "介绍一下大语言模型"}
{"id": 2, "instruction": "Python 和 Java 的主要区别是什么？"}
{"id": 3, "instruction": "解释一下机器学习中的过拟合"}
EOF

echo "✓ 创建文件: examples/data/sample_input.jsonl"

echo ""
echo "📝 步骤 2: 创建输入解析函数..."

# 创建输入解析函数
cat > examples/data/my_input_parser.py << 'EOF'
def parse_input(item):
    """将输入数据转换为 messages 格式"""
    return [
        {"role": "system", "content": "你是一个有帮助的AI助手。请简洁准确地回答问题。"},
        {"role": "user", "content": item["instruction"]}
    ]
EOF

echo "✓ 创建文件: examples/data/my_input_parser.py"

echo ""
echo "📝 步骤 3: 创建输出解析函数..."

# 创建输出解析函数
cat > examples/data/my_output_parser.py << 'EOF'
def parse_output(original_item, response):
    """处理模型输出"""
    return {
        "id": original_item["id"],
        "instruction": original_item["instruction"],
        "response": response,
        "length": len(response)
    }
EOF

echo "✓ 创建文件: examples/data/my_output_parser.py"

echo ""
echo "=================================="
echo "✅ 准备工作完成！"
echo "=================================="

echo ""
echo "🚀 现在可以运行推理了："
echo ""
echo "python inference.py \\"
echo "    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\"
echo "    --input_file examples/data/sample_input.jsonl \\"
echo "    --output_file examples/data/sample_output.jsonl \\"
echo "    --input_parser examples/data/my_input_parser.py \\"
echo "    --input_parser_fn parse_input \\"
echo "    --output_parser examples/data/my_output_parser.py \\"
echo "    --output_parser_fn parse_output \\"
echo "    --num_gpus 1 \\"
echo "    --tensor_parallel_size 1 \\"
echo "    --batch_size 8 \\"
echo "    --temperature 0.7 \\"
echo "    --max_tokens 512"
echo ""
echo "=================================="

