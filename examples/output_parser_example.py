"""
输出解析函数示例

输出解析函数的作用:
    接收原始输入数据项和模型生成的响应，返回最终要保存的数据

函数签名:
    def process_output(original_item: dict, response: str) -> dict:
        ...
        return result_item

参数:
    original_item: 原始输入数据项（字典）
    response: 模型生成的响应文本

返回:
    result_item: 要保存到输出文件的数据项（字典）
"""

from typing import Dict, Any
import json


def process_output(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 1: 简单追加响应
    
    输入:
        original_item = {"instruction": "介绍一下 AI"}
        response = "AI 是人工智能的缩写..."
    
    输出:
        {
            "instruction": "介绍一下 AI",
            "response": "AI 是人工智能的缩写..."
        }
    """
    return {
        **original_item,
        "response": response
    }


def process_with_metadata(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 2: 添加元数据
    
    输出包含原始数据、响应和额外的元数据
    """
    return {
        **original_item,
        "response": response,
        "response_length": len(response),
        "word_count": len(response.split())
    }


def process_qa_pair(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 3: 格式化为问答对
    
    输入:
        original_item = {"question": "什么是机器学习？", "context": "..."}
        response = "机器学习是..."
    
    输出:
        {
            "question": "什么是机器学习？",
            "answer": "机器学习是...",
            "context": "..."
        }
    """
    return {
        "question": original_item.get("question", ""),
        "answer": response,
        "context": original_item.get("context", "")
    }


def process_conversation(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 4: 构建对话历史
    
    将生成的响应添加到对话历史中
    """
    history = original_item.get("history", [])
    current_query = original_item.get("current_query", "")
    
    # 添加当前轮次的对话
    history.append({"role": "user", "content": current_query})
    history.append({"role": "assistant", "content": response})
    
    return {
        "id": original_item.get("id", ""),
        "conversation": history
    }


def process_with_instruction(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 5: 保留关键字段并重命名
    
    常用于指令微调数据集的构建
    """
    return {
        "instruction": original_item.get("instruction", ""),
        "input": original_item.get("input", ""),
        "output": response,
        "source": original_item.get("source", "generated")
    }


def process_extract_json(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 6: 从响应中提取 JSON
    
    如果模型输出是 JSON 格式，尝试解析它
    """
    try:
        # 尝试直接解析
        parsed_response = json.loads(response)
    except json.JSONDecodeError:
        # 如果失败，尝试提取 JSON 代码块
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed_response = json.loads(json_match.group(1))
            except:
                parsed_response = {"raw_text": response}
        else:
            parsed_response = {"raw_text": response}
    
    return {
        **original_item,
        "parsed_output": parsed_response,
        "raw_response": response
    }


def process_minimal(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 7: 最小化输出
    
    只保存必要的字段
    """
    return {
        "id": original_item.get("id", ""),
        "output": response
    }


def process_with_evaluation(original_item: dict, response: str) -> Dict[str, Any]:
    """
    示例 8: 添加简单评估指标
    
    可以在这里添加一些简单的质量检查
    """
    # 简单的质量检查
    is_valid = len(response) > 10 and not response.startswith("Error")
    
    return {
        **original_item,
        "response": response,
        "is_valid": is_valid,
        "response_stats": {
            "length": len(response),
            "has_numbers": any(c.isdigit() for c in response),
            "has_punctuation": any(c in "。！？,.!?" for c in response)
        }
    }

