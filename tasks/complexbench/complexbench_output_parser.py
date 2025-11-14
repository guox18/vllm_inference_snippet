"""
ComplexBench 数据集的输出解析函数
"""

from typing import Dict, Any


def parse_complexbench_output(original_item: dict, response: str) -> Dict[str, Any]:
    """
    将 ComplexBench 的推理结果格式化为指定格式

    输出格式:
        {
            "main_id": 0,
            "model": "Qwen2.5-7B-Instruct",
            "instruction": "...",
            "generated": "..."
        }

    参数:
        original_item: 原始输入数据项
        response: 模型生成的响应

    返回:
        result: 格式化的输出数据项
    """
    return {
        "main_id": original_item.get("main_id", 0),
        "model": "Qwen2.5-7B-Instruct",
        "instruction": original_item.get("instruction_en", ""),
        "generated": response,
    }
