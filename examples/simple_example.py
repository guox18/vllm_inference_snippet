"""
完整使用示例

这个示例展示了如何使用推理框架进行简单的文本生成任务
"""

# =============================================================================
# 步骤 1: 准备输入数据
# =============================================================================
# 创建示例输入文件: examples/sample_input.jsonl

sample_data = [
    {"id": 1, "instruction": "介绍一下大语言模型"},
    {"id": 2, "instruction": "Python 和 Java 的主要区别是什么？"},
    {"id": 3, "instruction": "解释一下机器学习中的过拟合"},
]

# 保存为 JSONL 文件
import json
with open('examples/sample_input.jsonl', 'w', encoding='utf-8') as f:
    for item in sample_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("✓ 创建示例输入文件: examples/sample_input.jsonl")


# =============================================================================
# 步骤 2: 创建输入解析函数
# =============================================================================
# 创建文件: examples/my_input_parser.py

input_parser_code = '''
def parse_input(item):
    """将输入数据转换为 messages 格式"""
    return [
        {"role": "system", "content": "你是一个有帮助的AI助手。请简洁准确地回答问题。"},
        {"role": "user", "content": item["instruction"]}
    ]
'''

with open('examples/my_input_parser.py', 'w', encoding='utf-8') as f:
    f.write(input_parser_code)

print("✓ 创建输入解析函数: examples/my_input_parser.py")


# =============================================================================
# 步骤 3: 创建输出解析函数
# =============================================================================
# 创建文件: examples/my_output_parser.py

output_parser_code = '''
def parse_output(original_item, response):
    """处理模型输出"""
    return {
        "id": original_item["id"],
        "instruction": original_item["instruction"],
        "response": response,
        "length": len(response)
    }
'''

with open('examples/my_output_parser.py', 'w', encoding='utf-8') as f:
    f.write(output_parser_code)

print("✓ 创建输出解析函数: examples/my_output_parser.py")


# =============================================================================
# 步骤 4: 运行推理
# =============================================================================
print("\n" + "="*60)
print("📋 运行推理的命令示例:")
print("="*60)

# 示例 1: 单卡推理
print("\n🔹 单卡推理:")
print("""
python inference.py \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --input_file examples/sample_input.jsonl \\
    --output_file examples/sample_output.jsonl \\
    --input_parser examples/my_input_parser.py \\
    --input_parser_fn parse_input \\
    --output_parser examples/my_output_parser.py \\
    --output_parser_fn parse_output \\
    --num_gpus 1 \\
    --tensor_parallel_size 1 \\
    --batch_size 8 \\
    --temperature 0.7 \\
    --max_tokens 512
""")

# 示例 2: 多卡张量并行
print("\n🔹 多卡张量并行 (TP=2，使用2张卡进行张量并行):")
print("""
python inference.py \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --input_file examples/sample_input.jsonl \\
    --output_file examples/sample_output.jsonl \\
    --input_parser examples/my_input_parser.py \\
    --input_parser_fn parse_input \\
    --output_parser examples/my_output_parser.py \\
    --output_parser_fn parse_output \\
    --num_gpus 2 \\
    --tensor_parallel_size 2 \\
    --batch_size 16
""")

# 示例 3: 数据并行 + 张量并行
print("\n🔹 数据并行 + 张量并行 (4张卡，TP=2，DP=2):")
print("""
torchrun --nproc_per_node=2 inference.py \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --input_file examples/sample_input.jsonl \\
    --output_file examples/sample_output.jsonl \\
    --input_parser examples/my_input_parser.py \\
    --input_parser_fn parse_input \\
    --output_parser examples/my_output_parser.py \\
    --output_parser_fn parse_output \\
    --num_gpus 4 \\
    --tensor_parallel_size 2 \\
    --batch_size 32
""")

# 示例 4: 断点续传
print("\n🔹 断点续传 (如果推理中断，可以从上次位置继续):")
print("""
python inference.py \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --input_file examples/sample_input.jsonl \\
    --output_file examples/sample_output.jsonl \\
    --input_parser examples/my_input_parser.py \\
    --input_parser_fn parse_input \\
    --output_parser examples/my_output_parser.py \\
    --output_parser_fn parse_output \\
    --num_gpus 1 \\
    --tensor_parallel_size 1 \\
    --resume
""")

# 示例 5: 调整推理参数
print("\n🔹 自定义推理参数:")
print("""
python inference.py \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --input_file examples/sample_input.jsonl \\
    --output_file examples/sample_output.jsonl \\
    --input_parser examples/my_input_parser.py \\
    --input_parser_fn parse_input \\
    --output_parser examples/my_output_parser.py \\
    --output_parser_fn parse_output \\
    --num_gpus 1 \\
    --tensor_parallel_size 1 \\
    --temperature 0.3 \\
    --top_p 0.95 \\
    --top_k 50 \\
    --max_tokens 1024 \\
    --batch_size 16 \\
    --gpu_memory_utilization 0.95
""")

print("\n" + "="*60)
print("✅ 准备工作完成！现在可以运行上述命令进行推理")
print("="*60)

