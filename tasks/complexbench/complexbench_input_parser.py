"""
ComplexBench 数据集的输入解析函数
"""

from typing import List, Dict


def parse_complexbench(item: dict) -> List[Dict[str, str]]:
    """
    将 ComplexBench 数据转换为 messages 格式

    参数:
        item: ComplexBench 数据项，包含 instruction 字段

    返回:
        messages: 适用于 Qwen 模型的消息列表
    """
    instruction = item.get("instruction_en", "")

    # 使用简单的 system prompt，让模型专注于任务本身
    return [
        {"role": "user", "content": instruction},
    ]
