"""
输入解析函数示例

输入解析函数的作用:
    从原始数据项中提取出模型推理所需的 messages 格式

函数签名:
    def parse_to_messages(item: dict) -> List[dict]:
        ...
        return messages

参数:
    item: 原始数据项（字典）

返回:
    messages: 符合聊天模板的消息列表，例如:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "用户的问题"}
        ]
"""

from typing import List, Dict


def parse_to_messages(item: dict) -> List[Dict[str, str]]:
    """
    示例 1: 简单的指令格式
    
    输入数据格式:
        {"instruction": "请介绍一下大语言模型"}
    
    输出:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请介绍一下大语言模型"}
        ]
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": item["instruction"]}
    ]


def parse_qa_format(item: dict) -> List[Dict[str, str]]:
    """
    示例 2: 问答格式
    
    输入数据格式:
        {"question": "什么是机器学习？", "context": "机器学习是..."}
    """
    content = f"背景信息：{item['context']}\n\n问题：{item['question']}"
    return [
        {"role": "user", "content": content}
    ]


def parse_with_history(item: dict) -> List[Dict[str, str]]:
    """
    示例 3: 带历史对话的格式
    
    输入数据格式:
        {
            "history": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
            ],
            "current_query": "介绍一下 Python"
        }
    """
    messages = item["history"].copy()
    messages.append({"role": "user", "content": item["current_query"]})
    return messages


def parse_with_custom_system(item: dict) -> List[Dict[str, str]]:
    """
    示例 4: 自定义系统提示词
    
    输入数据格式:
        {
            "system_prompt": "你是一个专业的代码助手",
            "user_input": "如何优化 Python 代码性能？"
        }
    """
    return [
        {"role": "system", "content": item["system_prompt"]},
        {"role": "user", "content": item["user_input"]}
    ]


def parse_multi_turn(item: dict) -> List[Dict[str, str]]:
    """
    示例 5: 多轮对话格式
    
    输入数据格式:
        {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好！"},
                {"from": "human", "value": "介绍一下自己"}
            ]
        }
    """
    messages = []
    role_mapping = {"human": "user", "gpt": "assistant"}
    
    for conv in item["conversations"]:
        messages.append({
            "role": role_mapping.get(conv["from"], conv["from"]),
            "content": conv["value"]
        })
    
    return messages

